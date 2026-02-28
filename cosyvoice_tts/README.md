# CosyVoice2 TTS 服务 — 多 GPU + Nginx 负载均衡

## 架构

```
客户端 ──wss──→ Nginx (least_conn) ──→ GPU 0  :9000  (独立进程)
                                   ├─→ GPU 1  :9001
                                   ├─→ GPU 2  :9002
                                   ├─→ ...
                                   └─→ GPU 6  :9006
```

每张卡一个独立 WebSocket 进程，无共享状态，Nginx 负载均衡。
简单、可靠、低延迟。

## 文件

```
cosyvoice_tts/
├── server.py     # 单卡 WebSocket TTS 服务
├── client.py     # 测试客户端 + 压测工具
├── launch.sh     # 多卡一键启动/停止/状态
└── README.md
```

## 快速开始

### 1. 启动所有 GPU

```bash
chmod +x cosyvoice_tts/launch.sh

# 自动检测所有 GPU，native 后端
./cosyvoice_tts/launch.sh start

# 指定 GPU + vllm 后端
./cosyvoice_tts/launch.sh start --gpus 0,1,2,3,4,5,6 --backend vllm --fp16

# 查看状态
./cosyvoice_tts/launch.sh status
```

### 2. 配置 Nginx

```bash
# 自动生成配置
./cosyvoice_tts/launch.sh nginx > /etc/nginx/conf.d/tts.conf
nginx -t && nginx -s reload
```

生成的配置：
```nginx
upstream tts_backend {
    least_conn;
    server 127.0.0.1:9000 max_fails=3 fail_timeout=10s;    # GPU 0
    server 127.0.0.1:9001 max_fails=3 fail_timeout=10s;    # GPU 1
    server 127.0.0.1:9002 max_fails=3 fail_timeout=10s;    # GPU 2
    server 127.0.0.1:9003 max_fails=3 fail_timeout=10s;    # GPU 3
    server 127.0.0.1:9004 max_fails=3 fail_timeout=10s;    # GPU 4
    server 127.0.0.1:9005 max_fails=3 fail_timeout=10s;    # GPU 5
    server 127.0.0.1:9006 max_fails=3 fail_timeout=10s;    # GPU 6
}

server {
    listen 8080;
    location /ws {
        proxy_pass http://tts_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 300s;
        proxy_buffering off;
    }
}
```

### 3. 测试

```bash
# 单次合成
python -m cosyvoice_tts.client --text "你好，欢迎使用。"

# 模拟 LLM 流式输出
python -m cosyvoice_tts.client --simulate-stream

# 打断测试
python -m cosyvoice_tts.client --test-interrupt

# 压测: 20 并发 × 5 轮
python -m cosyvoice_tts.client --bench --parallel 20 --rounds 5
```

## 动态扩缩容

```bash
# 加一张卡（新启动一个进程即可）
python -m cosyvoice_tts.server --gpu 7 --port 9007 --backend native &

# 然后更新 Nginx upstream 加一行:
#   server 127.0.0.1:9007;
nginx -s reload

# 下线一张卡
kill $(cat pids/gpu6.pid)
# 从 Nginx upstream 删除对应行
nginx -s reload
```

## 推理后端选择

| | native | vllm |
|---|---|---|
| 音色 | ✅ 与原模型一致 | ⚠️ 可能有差异 |
| 吞吐 | 较低 | 高 |
| 建议并发/卡 | 2-4 | 6-8 |
| vllm 依赖 | 不需要 | 需要安装 |
| 适用场景 | 音质优先 | 吞吐优先 |

可以混合部署：部分卡用 native，部分用 vllm。

## Systemd 服务化

```ini
# /etc/systemd/system/tts@.service
[Unit]
Description=CosyVoice TTS GPU %i
After=network.target

[Service]
Type=simple
User=admin
WorkingDirectory=/opt/cosyvoice
Environment=CUDA_VISIBLE_DEVICES=%i
ExecStart=/opt/cosyvoice/venv/bin/python -m cosyvoice_tts.server --gpu %i --port 900%i --backend native
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启动 7 张卡
for i in $(seq 0 6); do systemctl enable --now tts@$i; done

# 动态加卡
systemctl start tts@7

# 下线
systemctl stop tts@6
```
