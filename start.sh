#!/usr/bin/env bash
set -euo pipefail

echo "==================================="
echo "Starting ComfyUI Setup"
echo "==================================="

# Fix DNS resolution issues
echo "[network] Checking DNS configuration..."
echo "[network] Current /etc/resolv.conf:"
cat /etc/resolv.conf

# Add Google DNS if not present
if ! grep -q "8.8.8.8" /etc/resolv.conf 2>/dev/null; then
  echo "[network] Adding Google DNS to /etc/resolv.conf..."
  {
    echo "nameserver 8.8.8.8"
    echo "nameserver 8.8.4.4"
    echo "nameserver 1.1.1.1"
  } >> /etc/resolv.conf
  echo "[network] Updated /etc/resolv.conf:"
  cat /etc/resolv.conf
fi

# Test DNS resolution with multiple methods
echo "[network] Testing DNS resolution..."
if command -v nslookup >/dev/null 2>&1; then
  if nslookup pypi.org >/dev/null 2>&1; then
    echo "[network] DNS resolution working (nslookup test passed)"
  else
    echo "[network] WARNING: nslookup test failed"
  fi
fi

if command -v ping >/dev/null 2>&1; then
  if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1; then
    echo "[network] Basic connectivity OK (ping 8.8.8.8 successful)"
  else
    echo "[network] WARNING: Cannot ping 8.8.8.8"
  fi
fi

# Try to resolve pypi.org using getent
if command -v getent >/dev/null 2>&1; then
  if getent hosts pypi.org >/dev/null 2>&1; then
    echo "[network] DNS working (getent hosts pypi.org successful)"
  else
    echo "[network] WARNING: getent hosts pypi.org failed"
  fi
fi

# Wait for network to be ready before continuing (critical for RunPod timing issues)
echo "[network] Waiting for network readiness..."
MAX_WAIT=30
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
  if ping -c 1 -W 2 8.8.8.8 >/dev/null 2>&1 && \
     (getent hosts pypi.org >/dev/null 2>&1 || nslookup pypi.org >/dev/null 2>&1); then
    echo "[network] Network is ready!"
    break
  fi
  WAIT_COUNT=$((WAIT_COUNT + 1))
  if [ $WAIT_COUNT -lt $MAX_WAIT ]; then
    echo "[network] Network not ready, waiting... ($WAIT_COUNT/$MAX_WAIT)"
    sleep 1
  else
    echo "[network] WARNING: Network still not ready after ${MAX_WAIT}s, proceeding anyway"
  fi
done

COMFY_DIR="${COMFYUI_PATH:-/workspace/ComfyUI}"
CUSTOM_NODES="${COMFY_DIR}/custom_nodes"
MODELS_DIR="${COMFY_DIR}/models"

# Persistent RunPod volume (set RUNPOD_VOLUME in template if you want)
PERSIST_DIR="${RUNPOD_VOLUME:-/workspace/runpod-slim}"

# Optional baked fallback (if /workspace is a mounted empty volume)
BAKED_DIR="${COMFYUI_BAKED:-/opt/ComfyUI}"

mkdir -p "$(dirname "$COMFY_DIR")" "$PERSIST_DIR"

# If ComfyUI is missing but baked exists, restore it (RunPod mount scenario)
if [ ! -f "${COMFY_DIR}/main.py" ] && [ -f "${BAKED_DIR}/main.py" ]; then
  echo "[setup] Restoring ComfyUI from ${BAKED_DIR} -> ${COMFY_DIR} (mount detected)"
  rm -rf "${COMFY_DIR}"
  cp -a "${BAKED_DIR}" "${COMFY_DIR}"
fi

if [ ! -f "${COMFY_DIR}/main.py" ]; then
  echo "[fatal] ComfyUI not found at ${COMFY_DIR}. Check Dockerfile build or volume mount."
  exit 1
fi

mkdir -p "${CUSTOM_NODES}" "${MODELS_DIR}"

# -----------------------------
# Speed: persistent pip cache
# -----------------------------
export PIP_CACHE_DIR="${PERSIST_DIR}/.cache/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$PIP_CACHE_DIR"

# -----------------------------
# Hard constraints (prevents numpy2 / transformers drift)
# -----------------------------
CONSTRAINTS_FILE="${PERSIST_DIR}/pip-constraints.txt"
cat > "$CONSTRAINTS_FILE" <<'EOF'
numpy<2
protobuf<5
transformers>=4.45.0
safetensors
mediapipe==0.10.14
sageattention
EOF

export PIP_CONSTRAINT="$CONSTRAINTS_FILE"

echo "[pip] Enforcing constraints:"
cat "$CONSTRAINTS_FILE"

# Only install if versions are wrong (skip if already correct)
SKIP_PIP_INSTALL=0
python3 - <<'PY' && SKIP_PIP_INSTALL=1 || true
import sys
from packaging import version
try:
    import numpy
    import transformers
    import mediapipe
    assert numpy.__version__.startswith('1.')
    assert version.parse(transformers.__version__) >= version.parse('4.45.0')
    assert mediapipe.__version__ == '0.10.14'
    sys.exit(0)
except:
    sys.exit(1)
PY

if [ "$SKIP_PIP_INSTALL" = "0" ]; then
  echo "[pip] Installing/updating core dependencies..."

  # Retry pip install up to 3 times with exponential backoff
  MAX_RETRIES=3
  RETRY_COUNT=0
  RETRY_DELAY=5

  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "[pip] Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES..."

    if pip install -q --upgrade --prefer-binary \
      --retries 5 \
      --timeout 60 \
      -c "$CONSTRAINTS_FILE" \
      "numpy<2" \
      "protobuf<5" \
      "transformers>=4.45.0" \
      "safetensors" \
      "mediapipe==0.10.14" \
      "sageattention"; then
      echo "[pip] Core dependencies installed successfully"
      break
    else
      RETRY_COUNT=$((RETRY_COUNT + 1))
      if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo "[pip] Install failed, retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
        RETRY_DELAY=$((RETRY_DELAY * 2))
      else
        echo "[pip] WARNING: Failed to install some dependencies after $MAX_RETRIES attempts"
        echo "[pip] Continuing anyway - some features may not work"
      fi
    fi
  done
else
  echo "[pip] Core dependencies already correct, skipping install"
fi

echo "[debug] Versions:"
python3 - <<'PY'
import sys
print("python:", sys.version)
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
import numpy
print("numpy:", numpy.__version__)
import transformers
print("transformers:", transformers.__version__)
import mediapipe
print("mediapipe:", getattr(mediapipe, "__version__", "unknown"), "solutions:", hasattr(mediapipe, "solutions"))
try:
    import sageattention
    print("sageattention:", getattr(sageattention, "__version__", "installed (no version)"))
except ImportError:
    print("sageattention: NOT INSTALLED")
PY

# -----------------------------
# Helpers
# -----------------------------
download() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"

  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "[models] exists: $out"
    return 0
  fi

  echo "[models] downloading: $out"

  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 16 -s 16 -k 1M \
      --allow-overwrite=true \
      --file-allocation=none \
      --max-tries=5 \
      --retry-wait=3 \
      --connect-timeout=30 \
      --timeout=60 \
      --max-connection-per-server=16 \
      --min-split-size=1M \
      --split=16 \
      --stream-piece-selector=geom \
      --optimize-concurrent-downloads=true \
      -d "$(dirname "$out")" -o "$(basename "$out")" \
      "$url"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 8 --retry-delay 2 -C - -o "$out" "$url"
  else
    wget -c -O "$out" "$url"
  fi
}

civit_download() {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"

  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "[civitai] exists: $out"
    return 0
  fi

  echo "[civitai] downloading: $out"

  if command -v aria2c >/dev/null 2>&1; then
    local aria_opts=(
      -c -x 16 -s 16 -k 1M
      --allow-overwrite=true
      --file-allocation=none
      --max-tries=10
      --retry-wait=2
      --connect-timeout=30
      --timeout=60
      --max-connection-per-server=16
      --min-split-size=1M
      --split=16
      --stream-piece-selector=geom
      --optimize-concurrent-downloads=true
      -d "$(dirname "$out")" -o "$(basename "$out")"
    )

    if [ -n "${CIVITAI_TOKEN:-}" ]; then
      aria_opts+=( --header="Authorization: Bearer ${CIVITAI_TOKEN}" )
    fi

    aria2c "${aria_opts[@]}" "$url"
  else
    local header=()
    if [ -n "${CIVITAI_TOKEN:-}" ]; then
      header+=( -H "Authorization: Bearer ${CIVITAI_TOKEN}" )
    fi

    curl -L --fail --retry 10 --retry-delay 2 -C - \
      "${header[@]}" \
      -o "$out" "$url"
  fi

  # If we got HTML (login page), delete it so you dont think its a model
  if command -v file >/dev/null 2>&1 && file "$out" | grep -qi "HTML"; then
    echo "[civitai] ERROR: got HTML instead of model (token missing/invalid/gated). Removing $out"
    rm -f "$out"
    return 1
  fi
}

env_lora_download() {
  local url_var="$1"      # env var name, e.g. CHAR_LORA_URL
  local filename="${2:-}" # optional output filename override
  local out_dir="${MODELS_DIR}/loras"

  local url="${!url_var:-}"
  if [ -z "$url" ]; then
    echo "[lora] env ${url_var} is empty -> skip"
    return 0
  fi

  mkdir -p "$out_dir"

  # Auto-name from URL if not provided
  if [ -z "$filename" ]; then
    filename="$(basename "${url%%\?*}")"
  fi

  # sanitize spaces just in case
  filename="${filename// /_}"

  local out="${out_dir}/${filename}"

  echo "[lora] url_var=${url_var}"
  echo "[lora] url=${url}"
  echo "[lora] out=${out}"

  if [ -f "$out" ] && [ -s "$out" ]; then
    echo "[lora] exists: $out"
    return 0
  fi

  # Dropbox can be picky; use aria2c or curl with UA + retries + resume
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 16 -s 16 -k 1M \
      --allow-overwrite=true \
      --file-allocation=none \
      --max-tries=10 \
      --retry-wait=2 \
      --connect-timeout=30 \
      --timeout=60 \
      --max-connection-per-server=16 \
      --min-split-size=1M \
      --split=16 \
      --stream-piece-selector=geom \
      --optimize-concurrent-downloads=true \
      --user-agent="Mozilla/5.0" \
      -d "$(dirname "$out")" -o "$(basename "$out")" \
      "$url"
  else
    curl -L --fail --retry 10 --retry-delay 2 -C - \
      -A "Mozilla/5.0" \
      -o "$out" "$url"
  fi

  # Detect HTML instead of a safetensors binary
  if command -v file >/dev/null 2>&1 && file "$out" | grep -qi "HTML"; then
    echo "[lora] ERROR: got HTML instead of model (Dropbox auth/blocked). Removing $out"
    rm -f "$out"
    return 1
  fi

  echo "[lora] done: $(ls -lh "$out" | awk '{print $5, $9}')"
}

# Install node requirements but never allow torch stack / numpy / transformers to be changed.
safe_pip_install_req() {
  local req="$1"
  [ -f "$req" ] || return 0

  # Filter lines that must never override global pins
  local tmpreq
  tmpreq="$(mktemp)"
  grep -viE '^(torch|torchvision|torchaudio|numpy|transformers|tokenizers|protobuf)([<=> ].*)?$' "$req" > "$tmpreq" || true

  # Retry with exponential backoff
  local retries=3
  local delay=2
  for ((i=1; i<=retries; i++)); do
    if pip install -q --prefer-binary --retries 5 --timeout 60 \
      -c "$CONSTRAINTS_FILE" -r "$tmpreq"; then
      rm -f "$tmpreq"
      return 0
    fi
    if [ $i -lt $retries ]; then
      echo "  [pip] Retry $i/$retries failed, waiting ${delay}s..."
      sleep $delay
      delay=$((delay * 2))
    fi
  done

  echo "  [pip] WARNING: Failed to install requirements from $req"
  rm -f "$tmpreq"
  return 0
}

# -----------------------------
# Model directories
# -----------------------------
mkdir -p \
  "${MODELS_DIR}/sams" \
  "${MODELS_DIR}/ultralytics/bbox" \
  "${MODELS_DIR}/ultralytics/segm" \
  "${MODELS_DIR}/diffusion_models" \
  "${MODELS_DIR}/vae" \
  "${MODELS_DIR}/clip" \
  "${MODELS_DIR}/loras" \
  "${MODELS_DIR}/checkpoints" \
  "${MODELS_DIR}/SEEDVR2"

chmod -R 777 "${MODELS_DIR}/loras" || true

# -----------------------------
# Cache repos on persistent volume (DO THIS EARLY IN PARALLEL)
# -----------------------------
REPO_CACHE="${PERSIST_DIR}/_repos"
mkdir -p "$REPO_CACHE"

ZIT_REPO_DIR="${REPO_CACHE}/IIIIIIII_ZIT_V3"
ULTRA_REPO_DIR="${REPO_CACHE}/IIIIIIIII_ZIT_V3_Ultralytics"
BATCHNODE_REPO_DIR="${REPO_CACHE}/BatchnodeI9"
SAVEZIP_REPO_DIR="${REPO_CACHE}/savezipi9"
SEEDVR2_REPO_DIR="${REPO_CACHE}/ComfyUI-SeedVR2_VideoUpscaler"

UPDATE_NODES="${UPDATE_NODES:-0}"
UPDATE_MODELS="${UPDATE_MODELS:-0}"
UPDATE_BATCHNODE="${UPDATE_BATCHNODE:-0}"
UPDATE_SAVEZIP="${UPDATE_SAVEZIP:-0}"
UPDATE_SEEDVR2="${UPDATE_SEEDVR2:-0}"

# Clone/update all repos in parallel
echo "[repos] Setting up Git repositories in parallel..."

(
  if [ ! -d "${ZIT_REPO_DIR}/.git" ]; then
    echo "[nodes] cloning node pack into cache (one-time)..."
    rm -rf "${ZIT_REPO_DIR}"
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --shallow-submodules --recurse-submodules --progress \
      "https://github.com/rvspromotion-glitch/IIIIIIII_ZIT_V3.git" \
      "${ZIT_REPO_DIR}"
    git -C "${ZIT_REPO_DIR}" submodule update --init --recursive --depth 1 || true
  elif [ "$UPDATE_NODES" = "1" ]; then
    echo "[nodes] updating cached node pack..."
    git -C "${ZIT_REPO_DIR}" pull --rebase || true
    git -C "${ZIT_REPO_DIR}" submodule update --init --recursive || true
  else
    echo "[nodes] using cached node pack (no pull)"
  fi
) &

(
  if [ ! -d "${ULTRA_REPO_DIR}/.git" ]; then
    echo "[bbox] cloning ultralytics model repo (one-time)..."
    rm -rf "${ULTRA_REPO_DIR}"
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --progress \
      "https://github.com/rvspromotion-glitch/IIIIIIIII_ZIT_V3_Ultralytics.git" \
      "${ULTRA_REPO_DIR}"
  elif [ "$UPDATE_MODELS" = "1" ]; then
    echo "[bbox] updating ultralytics model repo..."
    git -C "${ULTRA_REPO_DIR}" pull --rebase || true
  else
    echo "[bbox] using cached ultralytics model repo (no pull)"
  fi
) &

(
  if [ ! -d "${BATCHNODE_REPO_DIR}/.git" ]; then
    echo "[nodes] cloning BatchnodeI9 into cache (one-time)..."
    rm -rf "${BATCHNODE_REPO_DIR}"
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --progress \
      "https://github.com/rvspromotion-glitch/BatchnodeI9.git" \
      "${BATCHNODE_REPO_DIR}"
  elif [ "$UPDATE_BATCHNODE" = "1" ]; then
    echo "[nodes] updating cached BatchnodeI9..."
    git -C "${BATCHNODE_REPO_DIR}" pull --rebase || true
  else
    echo "[nodes] using cached BatchnodeI9 (no pull)"
  fi
) &

(
  if [ ! -d "${SAVEZIP_REPO_DIR}/.git" ]; then
    echo "[nodes] cloning Save Image I9 into cache (one-time)..."
    rm -rf "${SAVEZIP_REPO_DIR}"
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --progress \
      "https://github.com/rvspromotion-glitch/savezipi9.git" \
      "${SAVEZIP_REPO_DIR}"
  elif [ "$UPDATE_SAVEZIP" = "1" ]; then
    echo "[nodes] updating cached Save Image I9..."
    git -C "${SAVEZIP_REPO_DIR}" pull --rebase || true
  else
    echo "[nodes] using cached Save Image I9 (no pull)"
  fi
) &

(
  if [ ! -d "${SEEDVR2_REPO_DIR}/.git" ]; then
    echo "[nodes] cloning SeedVR2 Video Upscaler into cache (one-time)..."
    rm -rf "${SEEDVR2_REPO_DIR}"
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --progress \
      "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git" \
      "${SEEDVR2_REPO_DIR}"
  elif [ "$UPDATE_SEEDVR2" = "1" ]; then
    echo "[nodes] updating cached SeedVR2 Video Upscaler..."
    git -C "${SEEDVR2_REPO_DIR}" pull --rebase || true
  else
    echo "[nodes] using cached SeedVR2 Video Upscaler (no pull)"
  fi
) &

# Wait for all repo operations
wait

echo "[repos] All repositories ready"

# Create symlinks immediately after repos are ready
echo "[nodes] creating symlinks in custom_nodes..."
for dir in "${ZIT_REPO_DIR}"/*; do
  [ -d "$dir" ] || continue
  node_name="$(basename "$dir")"
  case "$node_name" in .git|.github|__pycache__) continue ;; esac
  ln -sfn "$dir" "${CUSTOM_NODES}/${node_name}"
done

for dir in "${BATCHNODE_REPO_DIR}"/*; do
  [ -d "$dir" ] || continue
  node_name="$(basename "$dir")"
  case "$node_name" in .git|.github|__pycache__) continue ;; esac
  ln -sfn "$dir" "${CUSTOM_NODES}/${node_name}"
done

for dir in "${SAVEZIP_REPO_DIR}"/*; do
  [ -d "$dir" ] || continue
  node_name="$(basename "$dir")"
  case "$node_name" in .git|.github|__pycache__) continue ;; esac
  ln -sfn "$dir" "${CUSTOM_NODES}/${node_name}"
done

# Link SeedVR2 directly (it's a single node pack, not a collection)
ln -sfn "${SEEDVR2_REPO_DIR}" "${CUSTOM_NODES}/$(basename "${SEEDVR2_REPO_DIR}")"

# Sync bbox models
BBOX_DIR="${MODELS_DIR}/ultralytics/bbox"
BBOX_MARK="${PERSIST_DIR}/.bbox-models-copied"

if [ ! -f "$BBOX_MARK" ] || [ "$UPDATE_MODELS" = "1" ]; then
  echo "[bbox] syncing .pt files into ${BBOX_DIR}..."
  find "${ULTRA_REPO_DIR}" -type f -name "*.pt" -exec cp -f {} "${BBOX_DIR}/" \; || true
  touch "$BBOX_MARK"
fi

# -----------------------------
# Model downloads (ALL IN PARALLEL - maximum speed)
# -----------------------------
echo "[models] Downloading required models (fully parallel)..."

download "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  "${MODELS_DIR}/sams/sam_vit_b_01ec64.pth" &
download "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth" \
  "${MODELS_DIR}/sams/sam_vit_l_0b3195.pth" &

download "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/detection/bbox/face_yolov8m.pt" \
  "${MODELS_DIR}/ultralytics/bbox/face_yolov8m.pt" &
download "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt" \
  "${MODELS_DIR}/ultralytics/segm/person_yolov8m-seg.pt" &

download "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt" \
  "${MODELS_DIR}/ultralytics/bbox/yolov8n.pt" &
download "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt" \
  "${MODELS_DIR}/ultralytics/bbox/yolov8n-pose.pt" &
download "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt" \
  "${MODELS_DIR}/ultralytics/bbox/yolov8m.pt" &
download "https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt" \
  "${MODELS_DIR}/ultralytics/bbox/hand_yolov8n.pt" &

download "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt" \
  "${MODELS_DIR}/ultralytics/segm/yolov8n-seg.pt" &
download "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m-seg.pt" \
  "${MODELS_DIR}/ultralytics/segm/yolov8m-seg.pt" &

download "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors" \
  "${MODELS_DIR}/diffusion_models/z_image_turbo_bf16.safetensors" &
download "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors" \
  "${MODELS_DIR}/vae/ae.safetensors" &
download "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
  "${MODELS_DIR}/clip/qwen_3_4b.safetensors" &

civit_download "https://civitai.com/api/download/models/1511445?type=Model&format=SafeTensor" \
  "${MODELS_DIR}/loras/1511445_Spread i5XL.safetensors" &
civit_download "https://civitai.com/api/download/models/2435561?type=Model&format=SafeTensor&size=pruned&fp=fp16" \
  "${MODELS_DIR}/checkpoints/2435561_Photo4_fp16_pruned.safetensors" &

env_lora_download "CHAR_LORA_URL" &

# Wait for ALL downloads to complete
wait

echo "[models] Downloads completed."

# -----------------------------
# ⚠️ FIXED SECTION - Install node requirements from ALL repos
# -----------------------------
INSTALL_NODE_REQS="${INSTALL_NODE_REQS:-1}"
REQ_MARK="${PERSIST_DIR}/.node-reqs-installed"

if [ "$INSTALL_NODE_REQS" = "1" ]; then
  if [ ! -f "$REQ_MARK" ] || [ "$UPDATE_NODES" = "1" ] || [ "$UPDATE_BATCHNODE" = "1" ] || [ "$UPDATE_SAVEZIP" = "1" ] || [ "$UPDATE_SEEDVR2" = "1" ]; then
    echo "[pip] Installing node requirements (constrained)..."

    # Process ZIT node pack requirements
    for dir in "${ZIT_REPO_DIR}"/*; do
      [ -d "$dir" ] || continue
      req="${dir}/requirements.txt"
      if [ -f "$req" ]; then
        echo "  - [pip] ZIT/$(basename "$dir")/requirements.txt"
        safe_pip_install_req "$req"
      fi
    done

    # Process BatchnodeI9 requirements
    for dir in "${BATCHNODE_REPO_DIR}"/*; do
      [ -d "$dir" ] || continue
      req="${dir}/requirements.txt"
      if [ -f "$req" ]; then
        echo "  - [pip] BatchnodeI9/$(basename "$dir")/requirements.txt"
        safe_pip_install_req "$req"
      fi
    done

    # Process Save ZIP I9 requirements
    for dir in "${SAVEZIP_REPO_DIR}"/*; do
      [ -d "$dir" ] || continue
      req="${dir}/requirements.txt"
      if [ -f "$req" ]; then
        echo "  - [pip] Save-ZIP-I9/$(basename "$dir")/requirements.txt"
        safe_pip_install_req "$req"
      fi
    done

    # Process SeedVR2 requirements
    req="${SEEDVR2_REPO_DIR}/requirements.txt"
    if [ -f "$req" ]; then
      echo "  - [pip] SeedVR2/requirements.txt"
      safe_pip_install_req "$req"
    fi

    touch "$REQ_MARK"
  else
    echo "[pip] Node requirements already installed (skip)"
  fi
fi

# Final safety - ensure critical packages are correct
echo "[pip] Final safety check for critical packages..."
pip install -q --upgrade --prefer-binary --retries 5 --timeout 60 \
  -c "$CONSTRAINTS_FILE" "numpy<2" "mediapipe==0.10.14" || \
  echo "[pip] WARNING: Final safety check failed, continuing anyway"

# -----------------------------
# Start JupyterLab in background
# -----------------------------
echo "[jupyter] Starting JupyterLab..."
jupyter lab \
  --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
  --ServerApp.token='' --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir="${COMFY_DIR}" \
  >/workspace/jupyter.log 2>&1 &

echo "==================================="
echo "Launching ComfyUI"
echo "==================================="

cd "${COMFY_DIR}"
exec python3 main.py --listen 0.0.0.0 --port 8188
