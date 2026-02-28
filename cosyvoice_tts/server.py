"""
CosyVoice2 WebSocket TTS 单卡服务

每个进程绑定一张 GPU，独立提供 WebSocket 服务。
多进程 + Nginx 负载均衡实现高并发。

启动:
  python -m cosyvoice_tts.server --gpu 0 --port 9000 --backend native
  python -m cosyvoice_tts.server --gpu 1 --port 9001 --backend vllm --fp16

协议:
  客户端 → 服务端 (JSON text frame):
    {"event_name": "StartTTS", "prompt_text": "...", "prompt_wav": "./ref.wav"}
    {"event_name": "Text", "content": "你好，"}
    {"event_name": "StopTTS"}
    {"event_name": "Ping"}

  服务端 → 客户端:
    {"event_name": "TTSStarted", "session_id": "xxx", "sample_rate": 24000}   (JSON)
    [binary PCM int16 data]                                                     (binary)
    {"event_name": "TTSStopped", "session_id": "xxx", "reason": "complete|interrupted"}
    {"event_name": "Error", "message": "..."}
    {"event_name": "Pong"}
"""

import sys
import os
import json
import asyncio
import logging
import argparse
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.append('third_party/Matcha-TTS')

from cosyvoice.utils.common import set_all_random_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("tts-server")


# ============================================================
# 文本缓冲 & 断句
# ============================================================

SENTENCE_ENDINGS = set('。！？；\n!?;')
CLAUSE_ENDINGS = set('，,、：:—…')
MIN_SENTENCE_LEN = 5
MAX_BUFFER_LEN = 100


class TextBuffer:
    def __init__(self):
        self._buf = ""

    def add(self, text: str) -> list:
        self._buf += text
        sentences = []
        while True:
            sent = self._try_extract()
            if sent is None:
                break
            sentences.append(sent)
        return sentences

    def flush(self):
        if self._buf.strip():
            sent = self._buf.strip()
            self._buf = ""
            return sent
        self._buf = ""
        return None

    def _try_extract(self):
        for i, ch in enumerate(self._buf):
            if ch in SENTENCE_ENDINGS and i + 1 >= MIN_SENTENCE_LEN:
                sent = self._buf[:i + 1].strip()
                self._buf = self._buf[i + 1:]
                if sent:
                    return sent
        if len(self._buf) > MAX_BUFFER_LEN:
            for i in range(len(self._buf) - 1, MIN_SENTENCE_LEN - 1, -1):
                if self._buf[i] in CLAUSE_ENDINGS:
                    sent = self._buf[:i + 1].strip()
                    self._buf = self._buf[i + 1:]
                    if sent:
                        return sent
            sent = self._buf[:MAX_BUFFER_LEN].strip()
            self._buf = self._buf[MAX_BUFFER_LEN:]
            if sent:
                return sent
        return None


# ============================================================
# 音频工具
# ============================================================

def speech_to_pcm_bytes(tts_speech: torch.Tensor) -> bytes:
    audio = tts_speech.squeeze()
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak * 0.95
    pcm = (audio * 32767).clamp(-32768, 32767).to(torch.int16)
    return pcm.numpy().tobytes()


# ============================================================
# 模型加载
# ============================================================

_model = None
_executor = None


def load_model(args):
    global _model, _executor

    gpu_id = args.gpu
    backend = args.backend
    use_vllm = (backend == "vllm")

    log.info(f"[GPU {gpu_id}] 加载模型, backend={backend}")
    torch.cuda.set_device(gpu_id)

    if use_vllm:
        from vllm import ModelRegistry
        from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
        ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

    from cosyvoice.cli.cosyvoice import AutoModel

    _model = AutoModel(
        model_dir=args.model_dir,
        load_jit=True,
        load_trt=args.load_trt,
        load_vllm=use_vllm,
        fp16=args.fp16 if use_vllm else False,
        trt_concurrent=args.trt_concurrent if args.load_trt else 0,
    )

    _executor = ThreadPoolExecutor(
        max_workers=args.concurrency,
        thread_name_prefix=f"infer-gpu{gpu_id}",
    )

    log.info(f"[GPU {gpu_id}] 就绪: backend={backend}, vllm={use_vllm}, "
             f"fp16={args.fp16 if use_vllm else False}, sr={_model.sample_rate}")

    # 预热
    if args.warmup and os.path.exists(args.default_prompt_wav):
        log.info(f"[GPU {gpu_id}] 预热...")
        set_all_random_seed(42)
        for _ in _model.inference_zero_shot(
            "你好，欢迎使用。", "你好。", args.default_prompt_wav, stream=False
        ):
            pass
        for _ in _model.inference_zero_shot(
            "今天天气不错。", "你好。", args.default_prompt_wav, stream=True
        ):
            pass
        log.info(f"[GPU {gpu_id}] 预热完成")


def do_inference(text, prompt_text, prompt_wav, stream=True):
    """同步推理（在线程池中调用）"""
    results = []
    for _, r in enumerate(_model.inference_zero_shot(
        text, prompt_text, prompt_wav, stream=stream
    )):
        results.append(r)
    return results


# ============================================================
# 推理信号量 — 控制单卡并发
# ============================================================

_infer_semaphore = None


# ============================================================
# 合成会话
# ============================================================

class SynthSession:
    def __init__(self, session_id, prompt_text, prompt_wav):
        self.session_id = session_id
        self.prompt_text = prompt_text
        self.prompt_wav = prompt_wav
        self.text_buffer = TextBuffer()
        self.sentence_queue = asyncio.Queue()
        self.cancel_event = asyncio.Event()
        self.synth_done = asyncio.Event()
        self.worker_task = None

    def cancel(self):
        self.cancel_event.set()
        try:
            self.sentence_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass


# ============================================================
# WebSocket 连接处理
# ============================================================

# 统计
_stats = {"connections": 0, "total_served": 0, "total_sentences": 0}


async def handle_connection(ws):
    conn_id = uuid.uuid4().hex[:8]
    _stats["connections"] += 1
    _stats["total_served"] += 1
    log.info(f"[{conn_id}] 连接 (活跃={_stats['connections']})")

    current = None

    try:
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except Exception:
                await _send(ws, {"event_name": "Error", "message": "无效 JSON"})
                continue

            ev = msg.get("event_name")

            # ========== StartTTS ==========
            if ev == "StartTTS":
                if current and not current.synth_done.is_set():
                    current.cancel()
                    try:
                        await asyncio.wait_for(current.synth_done.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        if current.worker_task:
                            current.worker_task.cancel()
                    await _send(ws, {
                        "event_name": "TTSStopped",
                        "session_id": current.session_id,
                        "reason": "interrupted",
                    })

                prompt_text = msg.get("prompt_text", "")
                prompt_wav = msg.get("prompt_wav", "")
                if not prompt_wav or not os.path.exists(prompt_wav):
                    await _send(ws, {
                        "event_name": "Error",
                        "message": f"参考音频不存在: {prompt_wav}",
                    })
                    continue

                sid = uuid.uuid4().hex[:8]
                current = SynthSession(sid, prompt_text, prompt_wav)
                current.worker_task = asyncio.create_task(
                    _synth_worker(ws, conn_id, current)
                )
                await _send(ws, {
                    "event_name": "TTSStarted",
                    "session_id": sid,
                    "sample_rate": _model.sample_rate,
                })

            # ========== Text ==========
            elif ev == "Text":
                if not current or current.cancel_event.is_set():
                    await _send(ws, {"event_name": "Error", "message": "无活跃会话"})
                    continue
                content = msg.get("content", "")
                if content:
                    for s in current.text_buffer.add(content):
                        await current.sentence_queue.put(s)

            # ========== StopTTS ==========
            elif ev == "StopTTS":
                if not current or current.cancel_event.is_set():
                    continue
                remaining = current.text_buffer.flush()
                if remaining:
                    await current.sentence_queue.put(remaining)
                await current.sentence_queue.put(None)
                await current.synth_done.wait()
                await _send(ws, {
                    "event_name": "TTSStopped",
                    "session_id": current.session_id,
                    "reason": "complete",
                })

            elif ev == "Ping":
                await _send(ws, {"event_name": "Pong"})

            else:
                await _send(ws, {"event_name": "Error", "message": f"未知: {ev}"})

    except Exception as e:
        log.exception(f"[{conn_id}] 异常: {e}")
    finally:
        if current and not current.synth_done.is_set():
            current.cancel()
        _stats["connections"] -= 1
        log.info(f"[{conn_id}] 断开 (活跃={_stats['connections']})")


async def _send(ws, data):
    try:
        await ws.send(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


async def _synth_worker(ws, conn_id, session):
    loop = asyncio.get_event_loop()
    chunks = 0
    try:
        while not session.cancel_event.is_set():
            try:
                sentence = await asyncio.wait_for(session.sentence_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if sentence is None or session.cancel_event.is_set():
                break

            t0 = time.time()
            try:
                async with _infer_semaphore:
                    results = await loop.run_in_executor(
                        _executor,
                        do_inference,
                        sentence, session.prompt_text, session.prompt_wav, True,
                    )

                for r in results:
                    if session.cancel_event.is_set():
                        break
                    pcm = speech_to_pcm_bytes(r['tts_speech'])
                    await ws.send(pcm)
                    chunks += 1

                _stats["total_sentences"] += 1
                log.debug(f"[{conn_id}/{session.session_id}] "
                          f"'{sentence[:20]}' {time.time()-t0:.2f}s")

            except Exception as e:
                log.error(f"[{conn_id}] 合成失败: {e}")
                await _send(ws, {"event_name": "Error", "message": str(e)})

    except asyncio.CancelledError:
        pass
    finally:
        session.synth_done.set()


# ============================================================
# HTTP 健康检查 (可选，用于 Nginx 探活)
# ============================================================

async def start_health_server(host, port, gpu_id, backend):
    try:
        from aiohttp import web

        async def health(req):
            return web.json_response({
                "status": "ok",
                "gpu": gpu_id,
                "backend": backend,
                "connections": _stats["connections"],
                "total_served": _stats["total_served"],
                "total_sentences": _stats["total_sentences"],
            })

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_get("/stats", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        log.info(f"健康检查: http://{host}:{port}/health")
    except ImportError:
        log.info("aiohttp 未安装，跳过健康检查端点")


# ============================================================
# 主入口
# ============================================================

async def main(args):
    global _infer_semaphore

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    load_model(args)
    _infer_semaphore = asyncio.Semaphore(args.concurrency)

    await start_health_server(args.host, args.health_port, args.gpu, args.backend)

    import websockets
    log.info(f"ws://{args.host}:{args.port} | GPU={args.gpu} "
             f"backend={args.backend} concurrency={args.concurrency}")

    async with websockets.serve(
        handle_connection,
        args.host,
        args.port,
        max_size=10 * 1024 * 1024,
        ping_interval=30,
        ping_timeout=60,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice2 TTS Server (单卡)")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, required=True, help="WebSocket 端口")
    parser.add_argument("--health-port", type=int, default=None,
                        help="健康检查 HTTP 端口 (默认: port+1000)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--backend", choices=["vllm", "native"], default="native")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="最大并发推理数 (默认: native=3, vllm=8)")
    parser.add_argument("--model-dir", default="pretrained_models/CosyVoice2-0.5B")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--load-trt", action="store_true", default=False)
    parser.add_argument("--trt-concurrent", type=int, default=4)
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--default-prompt-wav", default="./asset/zero_shot_prompt.wav")

    args = parser.parse_args()

    if args.concurrency is None:
        args.concurrency = 8 if args.backend == "vllm" else 3
    if args.health_port is None:
        args.health_port = args.port + 1000

    asyncio.run(main(args))
