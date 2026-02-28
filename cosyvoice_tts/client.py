"""
WebSocket TTS 客户端（测试 + 压测）

用法:
  python -m cosyvoice_tts.client --text "你好，欢迎使用。"
  python -m cosyvoice_tts.client --simulate-stream
  python -m cosyvoice_tts.client --test-interrupt
  python -m cosyvoice_tts.client --bench --parallel 20 --rounds 5     # 压测
"""

import asyncio
import json
import argparse
import time
import wave
import statistics

import websockets


async def run_single(args, text=None, output=None):
    """单次 TTS 请求"""
    uri = f"ws://{args.host}:{args.port}/ws"
    text = text or args.text

    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        audio_chunks = []
        sample_rate = 24000
        done_event = asyncio.Event()
        first_audio_time = None
        t_start = time.time()

        async def recv():
            nonlocal sample_rate, first_audio_time
            while True:
                try:
                    msg = await ws.recv()
                except websockets.ConnectionClosed:
                    break
                if isinstance(msg, bytes):
                    if first_audio_time is None:
                        first_audio_time = time.time()
                    audio_chunks.append(msg)
                else:
                    data = json.loads(msg)
                    ev = data.get("event_name")
                    if ev == "TTSStarted":
                        sample_rate = data.get("sample_rate", 24000)
                    elif ev == "TTSStopped":
                        done_event.set()
                        return
                    elif ev == "Error":
                        print(f"Error: {data.get('message')}")

        recv_task = asyncio.create_task(recv())

        # StartTTS
        await ws.send(json.dumps({
            "event_name": "StartTTS",
            "prompt_text": args.prompt_text,
            "prompt_wav": args.prompt_wav,
        }))

        await asyncio.sleep(0.05)

        # 发送文本
        if args.simulate_stream:
            for i in range(0, len(text), args.chunk_size):
                await ws.send(json.dumps({
                    "event_name": "Text",
                    "content": text[i:i + args.chunk_size],
                }))
                await asyncio.sleep(args.interval)
        else:
            await ws.send(json.dumps({"event_name": "Text", "content": text}))

        await ws.send(json.dumps({"event_name": "StopTTS"}))
        await done_event.wait()
        recv_task.cancel()

    t_end = time.time()
    total_time = t_end - t_start
    first_chunk_latency = (first_audio_time - t_start) if first_audio_time else None

    # 保存
    if output and audio_chunks:
        pcm = b"".join(audio_chunks)
        with wave.open(output, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        duration = len(pcm) / 2 / sample_rate
        print(f"保存: {output} ({duration:.2f}s)")

    return {
        "total_time": total_time,
        "first_chunk_latency": first_chunk_latency,
        "audio_chunks": len(audio_chunks),
        "text_len": len(text),
    }


async def run_bench(args):
    """并发压测"""
    texts = [
        "你好，欢迎使用语音合成服务。",
        "今天天气真不错，我们一起出去走走吧。",
        "收到好友从远方寄来的生日礼物，那份意外的惊喜让我心中充满了甜蜜。",
        "这个产品的特点是高效、稳定、易用。",
        "请问您需要什么帮助吗？我可以为您解答。",
    ]

    print(f"压测: {args.parallel} 并发 × {args.rounds} 轮")
    print(f"目标: ws://{args.host}:{args.port}/ws")
    print("=" * 60)

    all_results = []

    for round_i in range(args.rounds):
        tasks = []
        for i in range(args.parallel):
            text = texts[i % len(texts)]
            tasks.append(run_single(args, text=text, output=None))

        t0 = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - t0

        ok = [r for r in results if isinstance(r, dict)]
        errors = [r for r in results if isinstance(r, Exception)]

        latencies = [r["first_chunk_latency"] for r in ok if r["first_chunk_latency"]]
        total_times = [r["total_time"] for r in ok]

        print(f"Round {round_i+1}: {len(ok)} ok, {len(errors)} err, "
              f"{elapsed:.2f}s, "
              f"首包={statistics.mean(latencies)*1000:.0f}ms" if latencies else "N/A",
              f"avg={statistics.mean(total_times):.2f}s" if total_times else "")

        all_results.extend(ok)

    # 汇总
    if all_results:
        all_lat = [r["first_chunk_latency"] for r in all_results if r["first_chunk_latency"]]
        all_total = [r["total_time"] for r in all_results]
        print("\n" + "=" * 60)
        print(f"总请求: {len(all_results)}")
        print(f"首包延迟: avg={statistics.mean(all_lat)*1000:.0f}ms, "
              f"p50={statistics.median(all_lat)*1000:.0f}ms, "
              f"p99={sorted(all_lat)[int(len(all_lat)*0.99)]*1000:.0f}ms")
        print(f"总耗时:  avg={statistics.mean(all_total):.2f}s")


async def run_interrupt_test(args):
    """打断测试"""
    uri = f"ws://{args.host}:{args.port}/ws"
    async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
        events = []

        async def recv():
            while True:
                try:
                    msg = await ws.recv()
                except websockets.ConnectionClosed:
                    break
                if isinstance(msg, bytes):
                    events.append(("audio", len(msg)))
                else:
                    data = json.loads(msg)
                    events.append((data.get("event_name"), data))
                    print(f"  ← {data.get('event_name')}: {data}")
                    if data.get("reason") == "complete":
                        return

        recv_task = asyncio.create_task(recv())

        # 第一轮
        print("→ 第一轮 StartTTS")
        await ws.send(json.dumps({
            "event_name": "StartTTS",
            "prompt_text": args.prompt_text,
            "prompt_wav": args.prompt_wav,
        }))
        long_text = "今天要给大家讲一个很长的故事，从前有座山山上有座庙。" * 5
        await ws.send(json.dumps({"event_name": "Text", "content": long_text}))
        await ws.send(json.dumps({"event_name": "StopTTS"}))

        await asyncio.sleep(1.0)

        # 打断
        print("→ 第二轮 StartTTS (打断!)")
        await ws.send(json.dumps({
            "event_name": "StartTTS",
            "prompt_text": args.prompt_text,
            "prompt_wav": args.prompt_wav,
        }))
        await asyncio.sleep(0.1)
        await ws.send(json.dumps({"event_name": "Text", "content": "打断后的新内容。"}))
        await ws.send(json.dumps({"event_name": "StopTTS"}))

        await recv_task
        audio_events = [e for e in events if e[0] == "audio"]
        print(f"\n收到 {len(audio_events)} 个音频包")


async def main(args):
    if args.bench:
        await run_bench(args)
    elif args.test_interrupt:
        await run_interrupt_test(args)
    else:
        result = await run_single(args, output=args.output)
        print(f"耗时: {result['total_time']:.2f}s, "
              f"首包: {result['first_chunk_latency']*1000:.0f}ms, "
              f"chunks: {result['audio_chunks']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080, help="Nginx 端口")
    parser.add_argument("--text", default="你好，欢迎使用语音合成服务。今天天气真不错。")
    parser.add_argument("--prompt-text", dest="prompt_text", default="希望你以后能够做的比我还好呦。")
    parser.add_argument("--prompt-wav", dest="prompt_wav", default="./asset/zero_shot_prompt.wav")
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--simulate-stream", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--test-interrupt", action="store_true")

    # 压测
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--parallel", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)

    args = parser.parse_args()
    asyncio.run(main(args))
