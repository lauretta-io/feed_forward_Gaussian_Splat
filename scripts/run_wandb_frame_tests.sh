#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PYTHON="${PYTHON:-.venv/bin/python}"
WANDB_PROJECT="${WANDB_PROJECT:-resplat-tests}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SCENE_NAME="${SCENE_NAME:-02267acf6fb98de36173bf4e7db9734c8c421dcb00267e42964dc15134cbb1be}"
COLMAP_DATA_DIR="${COLMAP_DATA_DIR:-datasets/dl3dv-colmap-demo}"
COLMAP_IMAGE_DIR="$COLMAP_DATA_DIR/$SCENE_NAME/images_4"

prepare_anysplat_input() {
  local count="$1"
  local out_dir="outputs/frame_tests/anysplat_inputs_${count}f"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  mapfile -t images < <(find "$COLMAP_IMAGE_DIR" -maxdepth 1 -type f -name '*.png' | sort | head -n "$count")
  if [[ "${#images[@]}" -ne "$count" ]]; then
    echo "Expected $count images in $COLMAP_IMAGE_DIR, found ${#images[@]}." >&2
    return 1
  fi

  for image in "${images[@]}"; do
    ln -s "$(realpath "$image")" "$out_dir/$(basename "$image")"
  done
}

run_resplat_colmap() {
  local count="$1"
  local context="$2"
  local targets="$3"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" WANDB_SILENT=true "$PYTHON" scripts/infer_colmap.py \
    --model_preset dl3dv_8v_256x448 \
    --data_dir "$COLMAP_DATA_DIR" \
    --scene_name "$SCENE_NAME" \
    --output_dir "outputs/frame_tests/resplat_colmap_${count}f" \
    --start_frame 0 \
    --frame_distance "$count" \
    --num_context "$context" \
    --num_target "$targets" \
    --target_selection remaining \
    --max_save_images 1 \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "frame-tests/resplat-colmap-${count}f"
}

run_mvsplat_re10k() {
  local count="$1"
  local gap=$((count - 1))
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" WANDB_SILENT=true PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" -m src.main \
    +experiment=mvsplat_re10k \
    mode=test \
    checkpointing.load=checkpoints/re10k.ckpt \
    dataset.highres=true \
    dataset.view_sampler.min_distance_between_context_views="$gap" \
    dataset.view_sampler.max_distance_between_context_views="$gap" \
    data_loader.test.num_workers=0 \
    test.eval_time_skip_steps=0 \
    test.compute_scores=true \
    test.save_image=true \
    test.save_video=false \
    test.save_gt_image=false \
    test.save_input_images=false \
    wandb.project="$WANDB_PROJECT" \
    wandb.name="frame-tests/mvsplat-re10k-${count}f" \
    'wandb.tags=[mvsplat,re10k,frame-test]' \
    output_dir="outputs/frame_tests/mvsplat_re10k_${count}f"
}

run_anysplat() {
  local count="$1"
  local input_dir="outputs/frame_tests/anysplat_inputs_${count}f"
  prepare_anysplat_input "$count"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" WANDB_SILENT=true "$PYTHON" scripts/infer_anysplat.py \
    --input-dir "$input_dir" \
    --output-dir "outputs/frame_tests/anysplat_${count}f" \
    --device cuda:0 \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "frame-tests/anysplat-${count}f"
}

run_resplat_colmap 5 4 1
run_resplat_colmap 10 8 2
run_mvsplat_re10k 5
run_mvsplat_re10k 10
run_anysplat 5
run_anysplat 10
