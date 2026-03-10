"""
CosyVoice2 WebSocket TTS 单卡服务（支持声音复刻）

每个进程绑定一张 GPU，独立提供 WebSocket 服务。
多进程 + Nginx 负载均衡实现高并发。

启动:
  python -m cosyvoice_tts.server --gpu 0 --port 9000 --backend native
  python -m cosyvoice_tts.server --gpu 1 --port 9001 --backend vllm --fp16

协议 (event_name):

  ═══════════════════════════════════════════════════════════════
  声音复刻（注册/管理说话人）
  ═══════════════════════════════════════════════════════════════

  注册声音 (上传参考音频 → 提取音色 → 持久化):
    {"event_name": "RegisterSpeaker",
     "speaker_id": "xiaoming",
     "prompt_text": "希望你以后能够做的比我还好呦。",
     "prompt_wav": "./voices/xiaoming.wav"}
  → {"event_name": "SpeakerRegistered", "speaker_id": "xiaoming"}

  删除声音:
    {"event_name": "DeleteSpeaker", "speaker_id": "xiaoming"}
  → {"event_name": "SpeakerDeleted", "speaker_id": "xiaoming"}

  列出已注册声音:
    {"event_name": "ListSpeakers"}
  → {"event_name": "SpeakerList", "speakers": ["xiaoming", "xiaohong"]}

  ═══════════════════════════════════════════════════════════════
  TTS 合成（支持多种模式）
  ═══════════════════════════════════════════════════════════════

  StartTTS 支持以下字段:
    {
      "event_name": "StartTTS",
      "mode": "zero_shot",           // 合成模式 (见下表)

      // --- zero_shot 模式 (默认) ---
      "prompt_text": "参考文本",
      "prompt_wav": "./ref.wav",

      // --- clone 模式 (使用已注册的声音) ---
      "speaker_id": "xiaoming",

      // --- cross_lingual 模式 ---
      "prompt_wav": "./ref.wav",

      // --- instruct 模式 ---
      "instruct_text": "用四川话说",
      "prompt_wav": "./ref.wav",
    }

  合成模式:
    zero_shot       零样本语音克隆 (需要 prompt_text + prompt_wav)
    clone           使用已注册的声音复刻 (需要 speaker_id)
    cross_lingual   跨语言合成 (需要 prompt_wav)
    instruct        指令控制 (需要 instruct_text + prompt_wav)

  Text / StopTTS / Ping 同之前不变。
"""

import sys
import os
import json
import asyncio
import logging
import argparse
import time
import uuid
import threading
import re
import struct
import hashlib
from concurrent.futures import ThreadPoolExecutor

import torch

sys.path.append('third_party/Matcha-TTS')

from cosyvoice.utils.common import set_all_random_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("tts-server")


# ============================================================
# 文本缓冲 & 断句
# ============================================================

SENTENCE_ENDINGS = set('。！？!?')
ALL_PUNCTUATION = set('。！？；，、：:—…!?;,.')
MIN_SENTENCE_LEN = 10
MIN_MERGE_LEN = 20
MAX_BUFFER_LEN = 100


class TextBuffer:
    def __init__(self):
        self._buf = ""
        self._first_cut = True  # 是否还未完成首次截取

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
            self._first_cut = True
            return sent
        self._buf = ""
        self._first_cut = True
        return None

    def _try_extract(self):
        if self._first_cut:
            # 首次：长度 > MIN_SENTENCE_LEN 且遇到任意标点即截断
            for i, ch in enumerate(self._buf):
                if ch in ALL_PUNCTUATION and i + 1 >= MIN_SENTENCE_LEN:
                    sent = self._buf[:i + 1].strip()
                    self._buf = self._buf[i + 1:]
                    self._first_cut = False
                    if sent:
                        return sent
        else:
            # 后续：只按句号/问号/感叹号截断
            # 如果截断后句子 < MIN_MERGE_LEN 字，则与后面的句子拼接
            for i, ch in enumerate(self._buf):
                if ch in SENTENCE_ENDINGS:
                    candidate = self._buf[:i + 1].strip()
                    if candidate and len(candidate) < MIN_MERGE_LEN:
                        # 太短，继续往后找下一个句号/问号/感叹号
                        continue
                    sent = self._buf[:i + 1].strip()
                    self._buf = self._buf[i + 1:]
                    if sent:
                        return sent

        # 兜底：缓冲区超长时强制截断
        if len(self._buf) > MAX_BUFFER_LEN:
            for i in range(len(self._buf) - 1, -1, -1):
                if self._buf[i] in SENTENCE_ENDINGS:
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


# 括号匹配：全角（）和半角()
_BRACKET_RE = re.compile(r'[（(]([^）)]*)[）)]')
# "空x秒" 模式
_SILENCE_RE = re.compile(r'^空(\d+(?:\.\d+)?)秒$')
# app 替换（大小写不敏感）
_APP_RE = re.compile(r'(?i)app')


def _generate_silence_pcm(seconds: float, sample_rate: int) -> bytes:
    """生成指定秒数的静音 PCM (16-bit, mono)"""
    num_samples = int(sample_rate * seconds)
    return b'\x00\x00' * num_samples


def preprocess_sentence(text: str):
    """
    预处理句子，返回 segments 列表。
    每个 segment 是 ('text', str) 或 ('silence', float_seconds)。
    - （空3秒）→ ('silence', 3.0)
    - 其他括号内容 → 删除（不合成）
    - app/APP/App → 替换为 "A批批"
    """
    segments = []
    last_end = 0
    for m in _BRACKET_RE.finditer(text):
        # 括号前的文本
        before = text[last_end:m.start()]
        if before.strip():
            segments.append(('text', _APP_RE.sub('A批批', before)))
        # 括号内容
        inner = m.group(1).strip()
        sm = _SILENCE_RE.match(inner)
        if sm:
            segments.append(('silence', float(sm.group(1))))
        # 其他括号内容：忽略（不合成）
        last_end = m.end()
    # 剩余文本
    remaining = text[last_end:]
    if remaining.strip():
        segments.append(('text', _APP_RE.sub('A批批', remaining)))
    # 如果没有括号，整句做 app 替换
    if not segments and text.strip():
        segments.append(('text', _APP_RE.sub('A批批', text)))
    return segments


# ============================================================
# PCM 缓存（磁盘）
# ============================================================

_cache_dir = None          # 初始化时设置
_cache_lock = threading.Lock()


def _cache_key(text: str, mode: str, speaker_id: str) -> str:
    """基于文本+模式+说话人生成缓存 key（sha256 前16字节hex）"""
    raw = f"{text}|{mode}|{speaker_id or ''}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]


def _cache_path(key: str) -> str:
    # 二级子目录防止单目录文件过多: ab/cd/abcd...pcm
    sub = os.path.join(_cache_dir, key[:2], key[2:4])
    return os.path.join(sub, key + '.pcm')


def cache_get(text: str, mode: str, speaker_id: str):
    """查缓存，命中返回 bytes，否则 None"""
    if _cache_dir is None:
        return None
    key = _cache_key(text, mode, speaker_id)
    path = _cache_path(key)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception:
            return None
    return None


def cache_put(text: str, mode: str, speaker_id: str, pcm_data: bytes):
    """写缓存"""
    if _cache_dir is None:
        return
    key = _cache_key(text, mode, speaker_id)
    path = _cache_path(key)
    with _cache_lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            with open(path, 'wb') as f:
                f.write(pcm_data)
        except Exception as e:
            log.warning(f"缓存写入失败: {e}")


# ============================================================
# 模型加载
# ============================================================

_model = None
_executor = None
_spk_lock = threading.Lock()  # 说话人注册/删除操作的线程锁


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

    # 加载已持久化的说话人信息
    spk_count = _load_saved_speakers()
    if spk_count > 0:
        log.info(f"[GPU {gpu_id}] 已加载 {spk_count} 个已注册说话人")

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


def _load_saved_speakers() -> int:
    """加载已持久化的说话人（从模型目录的 spk_info 文件）"""
    try:
        # CosyVoice 的 save_spkinfo/load_spkinfo 会在 model_dir 下保存
        spk_info_path = os.path.join(_model.model_dir, 'spk_info.json')
        if os.path.exists(spk_info_path):
            with open(spk_info_path, 'r') as f:
                info = json.load(f)
            return len(info) if isinstance(info, dict) else 0
    except Exception as e:
        log.warning(f"加载说话人信息失败: {e}")
    return 0


# ============================================================
# 推理函数 — 支持多种模式
# ============================================================

def do_inference(text, mode, prompt_text, prompt_wav,
                 speaker_id, instruct_text, stream=True):
    """同步推理，根据 mode 调用不同接口"""
    results = []

    if mode == "clone":
        # 使用已注册的说话人音色
        for _, r in enumerate(_model.inference_zero_shot(
            text, '', '', zero_shot_spk_id=speaker_id,
            stream=stream, text_frontend=True,
        )):
            results.append(r)

    elif mode == "cross_lingual":
        # 跨语言合成
        for _, r in enumerate(_model.inference_cross_lingual(
            text, prompt_wav, stream=stream,
        )):
            results.append(r)

    elif mode == "instruct":
        # 指令控制合成
        for _, r in enumerate(_model.inference_instruct2(
            text, instruct_text, prompt_wav,
            stream=stream,
        )):
            results.append(r)

    else:
        # zero_shot（默认）
        for _, r in enumerate(_model.inference_zero_shot(
            text, prompt_text, prompt_wav,
            stream=stream,
        )):
            results.append(r)

    return results


# ============================================================
# 说话人管理
# ============================================================

def register_speaker(speaker_id: str, prompt_text: str, prompt_wav: str) -> bool:
    """注册说话人（线程安全）"""
    with _spk_lock:
        ok = _model.add_zero_shot_spk(prompt_text, prompt_wav, speaker_id)
        if ok:
            _model.save_spkinfo()
        return ok


def delete_speaker(speaker_id: str) -> bool:
    """删除说话人"""
    with _spk_lock:
        try:
            if hasattr(_model.model, 'spk_info') and speaker_id in _model.model.spk_info:
                del _model.model.spk_info[speaker_id]
                _model.save_spkinfo()
                return True
            # 另一种存储方式
            if hasattr(_model, 'spk_info') and speaker_id in _model.spk_info:
                del _model.spk_info[speaker_id]
                _model.save_spkinfo()
                return True
        except Exception as e:
            log.error(f"删除说话人失败: {e}")
    return False


def list_speakers() -> list:
    """列出所有已注册的说话人"""
    try:
        if hasattr(_model.model, 'spk_info'):
            return list(_model.model.spk_info.keys())
        if hasattr(_model, 'spk_info'):
            return list(_model.spk_info.keys())
    except Exception:
        pass
    # 尝试从文件读取
    try:
        spk_info_path = os.path.join(_model.model_dir, 'spk_info.json')
        if os.path.exists(spk_info_path):
            with open(spk_info_path, 'r') as f:
                info = json.load(f)
            return list(info.keys()) if isinstance(info, dict) else []
    except Exception:
        pass
    return []


# ============================================================
# 推理信号量
# ============================================================

_infer_semaphore = None


# ============================================================
# 合成会话
# ============================================================

class SynthSession:
    def __init__(self, session_id, mode, prompt_text, prompt_wav,
                 speaker_id, instruct_text):
        self.session_id = session_id
        self.mode = mode
        self.prompt_text = prompt_text
        self.prompt_wav = prompt_wav
        self.speaker_id = speaker_id
        self.instruct_text = instruct_text
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

_stats = {"connections": 0, "total_served": 0, "total_sentences": 0, "cache_hits": 0}


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

            # ========== RegisterSpeaker ==========
            if ev == "RegisterSpeaker":
                speaker_id = msg.get("speaker_id", "").strip()
                prompt_text = msg.get("prompt_text", "")
                prompt_wav = msg.get("prompt_wav", "")

                if not speaker_id:
                    await _send(ws, {"event_name": "Error",
                                     "message": "speaker_id 不能为空"})
                    continue
                if not prompt_wav or not os.path.exists(prompt_wav):
                    await _send(ws, {"event_name": "Error",
                                     "message": f"参考音频不存在: {prompt_wav}"})
                    continue
                if not prompt_text:
                    await _send(ws, {"event_name": "Error",
                                     "message": "prompt_text 不能为空（需要与参考音频对应的文本）"})
                    continue

                loop = asyncio.get_event_loop()
                ok = await loop.run_in_executor(
                    _executor, register_speaker,
                    speaker_id, prompt_text, prompt_wav,
                )

                if ok:
                    log.info(f"[{conn_id}] 注册说话人: {speaker_id}")
                    await _send(ws, {
                        "event_name": "SpeakerRegistered",
                        "speaker_id": speaker_id,
                    })
                else:
                    await _send(ws, {"event_name": "Error",
                                     "message": f"注册说话人失败: {speaker_id}"})

            # ========== DeleteSpeaker ==========
            elif ev == "DeleteSpeaker":
                speaker_id = msg.get("speaker_id", "").strip()
                if not speaker_id:
                    await _send(ws, {"event_name": "Error",
                                     "message": "speaker_id 不能为空"})
                    continue

                loop = asyncio.get_event_loop()
                ok = await loop.run_in_executor(_executor, delete_speaker, speaker_id)
                if ok:
                    log.info(f"[{conn_id}] 删除说话人: {speaker_id}")
                    await _send(ws, {
                        "event_name": "SpeakerDeleted",
                        "speaker_id": speaker_id,
                    })
                else:
                    await _send(ws, {"event_name": "Error",
                                     "message": f"说话人不存在: {speaker_id}"})

            # ========== ListSpeakers ==========
            elif ev == "ListSpeakers":
                speakers = list_speakers()
                await _send(ws, {
                    "event_name": "SpeakerList",
                    "speakers": speakers,
                })

            # ========== StartTTS ==========
            elif ev == "StartTTS":
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

                mode = msg.get("mode", "zero_shot")
                prompt_text = msg.get("prompt_text", "")
                prompt_wav = msg.get("prompt_wav", "")
                speaker_id = msg.get("speaker_id", "")
                instruct_text = msg.get("instruct_text", "")

                # 参数校验
                err = _validate_start(mode, prompt_text, prompt_wav,
                                      speaker_id, instruct_text)
                if err:
                    await _send(ws, {"event_name": "Error", "message": err})
                    continue

                sid = uuid.uuid4().hex[:8]
                current = SynthSession(
                    sid, mode, prompt_text, prompt_wav,
                    speaker_id, instruct_text,
                )
                current.worker_task = asyncio.create_task(
                    _synth_worker(ws, conn_id, current)
                )
                await _send(ws, {
                    "event_name": "TTSStarted",
                    "session_id": sid,
                    "mode": mode,
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


def _validate_start(mode, prompt_text, prompt_wav, speaker_id, instruct_text):
    """校验 StartTTS 参数，返回错误信息或 None"""
    if mode == "zero_shot":
        if not prompt_wav or not os.path.exists(prompt_wav):
            return f"zero_shot 模式需要 prompt_wav，文件不存在: {prompt_wav}"
        if not prompt_text:
            return "zero_shot 模式需要 prompt_text（参考音频对应的文本）"
    elif mode == "clone":
        if not speaker_id:
            return "clone 模式需要 speaker_id"
        if speaker_id not in list_speakers():
            return f"说话人未注册: {speaker_id}，请先 RegisterSpeaker"
    elif mode == "cross_lingual":
        if not prompt_wav or not os.path.exists(prompt_wav):
            return f"cross_lingual 模式需要 prompt_wav，文件不存在: {prompt_wav}"
    elif mode == "instruct":
        if not prompt_wav or not os.path.exists(prompt_wav):
            return f"instruct 模式需要 prompt_wav，文件不存在: {prompt_wav}"
        if not instruct_text:
            return "instruct 模式需要 instruct_text"
    else:
        return f"不支持的合成模式: {mode}，可选: zero_shot/clone/cross_lingual/instruct"
    return None


async def _send(ws, data):
    try:
        await ws.send(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


async def _synth_worker(ws, conn_id, session: SynthSession):
    loop = asyncio.get_event_loop()
    chunks = 0
    try:
        while not session.cancel_event.is_set():
            try:
                sentence = await asyncio.wait_for(
                    session.sentence_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            if sentence is None or session.cancel_event.is_set():
                break

            t0 = time.time()
            try:
                segments = preprocess_sentence(sentence)
                for seg_type, seg_value in segments:
                    if session.cancel_event.is_set():
                        break
                    if seg_type == 'silence':
                        silence_pcm = _generate_silence_pcm(
                            seg_value, _model.sample_rate)
                        await ws.send(silence_pcm)
                        chunks += 1
                    elif seg_type == 'text' and seg_value.strip():
                        # 查缓存
                        cached = cache_get(seg_value, session.mode,
                                           session.speaker_id)
                        if cached:
                            await ws.send(cached)
                            chunks += 1
                            _stats["cache_hits"] += 1
                            log.debug(f"[{conn_id}] cache hit: "
                                      f"'{seg_value[:20]}'")
                        else:
                            async with _infer_semaphore:
                                results = await loop.run_in_executor(
                                    _executor,
                                    do_inference,
                                    seg_value, session.mode,
                                    session.prompt_text, session.prompt_wav,
                                    session.speaker_id, session.instruct_text,
                                    True,
                                )
                            all_pcm = bytearray()
                            for r in results:
                                if session.cancel_event.is_set():
                                    break
                                pcm = speech_to_pcm_bytes(r['tts_speech'])
                                all_pcm.extend(pcm)
                                await ws.send(pcm)
                                chunks += 1
                            # 写缓存
                            if all_pcm and not session.cancel_event.is_set():
                                cache_put(seg_value, session.mode,
                                          session.speaker_id, bytes(all_pcm))

                _stats["total_sentences"] += 1
                log.debug(f"[{conn_id}/{session.session_id}] "
                          f"[{session.mode}] '{sentence[:20]}' {time.time()-t0:.2f}s")

            except Exception as e:
                log.error(f"[{conn_id}] 合成失败: {e}")
                await _send(ws, {"event_name": "Error", "message": str(e)})

    except asyncio.CancelledError:
        pass
    finally:
        session.synth_done.set()


# ============================================================
# HTTP 健康检查
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
                "cache_hits": _stats["cache_hits"],
                "registered_speakers": list_speakers(),
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
    parser.add_argument("--cache-dir", default="./tts_cache",
                        help="PCM 缓存目录 (默认: ./tts_cache，设为空禁用)")

    args = parser.parse_args()

    if args.concurrency is None:
        args.concurrency = 8 if args.backend == "vllm" else 3
    if args.health_port is None:
        args.health_port = args.port + 1000

    # 初始化缓存目录
    global _cache_dir
    if args.cache_dir and args.cache_dir.strip():
        _cache_dir = os.path.abspath(args.cache_dir)
        os.makedirs(_cache_dir, exist_ok=True)
        log.info(f"PCM 缓存目录: {_cache_dir}")
    else:
        _cache_dir = None
        log.info("PCM 缓存已禁用")

    asyncio.run(main(args))
