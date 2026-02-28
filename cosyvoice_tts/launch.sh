#!/bin/bash
# ============================================================
# CosyVoice TTS 多 GPU 一键启动 / 停止脚本
#
# 用法:
#   ./launch.sh start                    # 启动所有 GPU (native)
#   ./launch.sh start --backend vllm     # 启动所有 GPU (vllm)
#   ./launch.sh start --gpus 0,1,2       # 指定 GPU
#   ./launch.sh stop                     # 停止所有
#   ./launch.sh status                   # 查看状态
#   ./launch.sh restart                  # 重启
# ============================================================

set -e

# ---- 配置 ----
BASE_PORT=9000          # 第一张卡的 WebSocket 端口，后续递增
MODEL_DIR="pretrained_models/CosyVoice2-0.5B"
BACKEND="native"        # native 或 vllm
CONCURRENCY=""          # 留空则自动 (native=3, vllm=8)
FP16=""                 # --fp16 (仅 vllm)
EXTRA_ARGS=""
LOG_DIR="logs"
PID_DIR="pids"
PROMPT_WAV="./asset/zero_shot_prompt.wav"

# ---- 解析参数 ----
ACTION="${1:-help}"
shift || true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)       GPUS="$2"; shift 2;;
        --backend)    BACKEND="$2"; shift 2;;
        --base-port)  BASE_PORT="$2"; shift 2;;
        --model-dir)  MODEL_DIR="$2"; shift 2;;
        --concurrency) CONCURRENCY="$2"; shift 2;;
        --fp16)       FP16="--fp16"; shift;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

# 自动检测 GPU 数量
if [ -z "$GPUS" ]; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    GPUS=$(seq -s, 0 $((GPU_COUNT - 1)))
fi

mkdir -p "$LOG_DIR" "$PID_DIR"

IFS=',' read -ra GPU_LIST <<< "$GPUS"

# ---- 函数 ----

do_start() {
    echo "启动 ${#GPU_LIST[@]} 个 TTS 服务 (backend=$BACKEND)"
    echo "================================================"

    for i in "${!GPU_LIST[@]}"; do
        gpu=${GPU_LIST[$i]}
        port=$((BASE_PORT + gpu))
        health_port=$((port + 1000))
        pid_file="$PID_DIR/gpu${gpu}.pid"
        log_file="$LOG_DIR/gpu${gpu}.log"

        # 检查是否已在运行
        if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
            echo "  GPU $gpu: 已在运行 (PID $(cat "$pid_file"), port $port)"
            continue
        fi

        cmd="python -m cosyvoice_tts.server \
            --gpu $gpu \
            --port $port \
            --health-port $health_port \
            --backend $BACKEND \
            --model-dir $MODEL_DIR \
            --default-prompt-wav $PROMPT_WAV \
            $( [ -n "$CONCURRENCY" ] && echo "--concurrency $CONCURRENCY" ) \
            $FP16 \
            $EXTRA_ARGS"

        echo "  GPU $gpu: port=$port, health=$health_port, log=$log_file"
        nohup $cmd > "$log_file" 2>&1 &
        echo $! > "$pid_file"
    done

    echo ""
    echo "Nginx upstream 配置:"
    echo "  upstream tts_backend {"
    for gpu in "${GPU_LIST[@]}"; do
        port=$((BASE_PORT + gpu))
        echo "      server 127.0.0.1:${port};    # GPU $gpu"
    done
    echo "  }"
    echo ""
    echo "启动完成! 使用 '$0 status' 查看状态"
}

do_stop() {
    echo "停止所有 TTS 服务..."
    for gpu in "${GPU_LIST[@]}"; do
        pid_file="$PID_DIR/gpu${gpu}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  GPU $gpu: 停止 PID $pid"
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pid_file"
        fi
    done
    echo "已停止"
}

do_status() {
    printf "%-6s %-8s %-8s %-8s %-10s %-10s\n" "GPU" "PORT" "PID" "STATUS" "BACKEND" "CONNS"
    printf "%s\n" "------------------------------------------------------------"

    for gpu in "${GPU_LIST[@]}"; do
        port=$((BASE_PORT + gpu))
        health_port=$((port + 1000))
        pid_file="$PID_DIR/gpu${gpu}.pid"
        pid="-"
        status="stopped"
        conns="-"

        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                status="running"
                # 尝试获取健康检查
                health=$(curl -s --connect-timeout 1 "http://127.0.0.1:${health_port}/health" 2>/dev/null || echo "")
                if [ -n "$health" ]; then
                    conns=$(echo "$health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('connections','-'))" 2>/dev/null || echo "-")
                fi
            else
                status="dead"
                rm -f "$pid_file"
            fi
        fi

        printf "%-6s %-8s %-8s %-8s %-10s %-10s\n" "$gpu" "$port" "$pid" "$status" "$BACKEND" "$conns"
    done
}

do_restart() {
    do_stop
    sleep 2
    do_start
}

# ---- 生成 Nginx 配置 ----
do_nginx_conf() {
    cat <<EOF
# /etc/nginx/conf.d/tts.conf
# CosyVoice TTS WebSocket 负载均衡

upstream tts_backend {
    # 最少连接数调度 — 低延迟优先
    least_conn;

EOF
    for gpu in "${GPU_LIST[@]}"; do
        port=$((BASE_PORT + gpu))
        echo "    server 127.0.0.1:${port} max_fails=3 fail_timeout=10s;    # GPU $gpu"
    done

    cat <<EOF
}

server {
    listen 8080;
    # listen 443 ssl;
    # ssl_certificate     /path/to/cert.pem;
    # ssl_certificate_key /path/to/key.pem;

    # WebSocket TTS
    location /ws {
        proxy_pass http://tts_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;

        # WebSocket 缓冲优化
        proxy_buffering off;
        proxy_buffer_size 8k;
    }

    # 健康检查 (内部)
    location /health {
        proxy_pass http://tts_backend;
    }
}
EOF
}

# ---- 主逻辑 ----
case "$ACTION" in
    start)    do_start;;
    stop)     do_stop;;
    status)   do_status;;
    restart)  do_restart;;
    nginx)    do_nginx_conf;;
    *)
        echo "CosyVoice TTS 多 GPU 管理工具"
        echo ""
        echo "用法: $0 <command> [options]"
        echo ""
        echo "命令:"
        echo "  start     启动服务"
        echo "  stop      停止服务"
        echo "  status    查看状态"
        echo "  restart   重启"
        echo "  nginx     生成 Nginx 配置"
        echo ""
        echo "选项:"
        echo "  --gpus 0,1,2     指定 GPU (默认: 自动检测所有)"
        echo "  --backend native 推理后端: native|vllm (默认: native)"
        echo "  --base-port 9000 起始端口 (默认: 9000)"
        echo "  --concurrency N  单卡并发数"
        echo "  --fp16           启用 FP16 (仅 vllm)"
        echo "  --model-dir PATH 模型路径"
        ;;
esac
