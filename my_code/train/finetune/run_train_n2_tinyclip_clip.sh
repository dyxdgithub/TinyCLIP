#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

n2_dir="${repo_root}/my_code/data/OpenImage/meta/Hierarchy/n2"
split_csv="${N2_SPLIT_CSV:-${n2_dir}/Image_IDs_n2_all_downloaded_split_8_1_1.csv}"
caption_csv="${N2_CAPTION_CSV:-${n2_dir}/Image IDs_with_last_two_layers_multi_members_n2_one_label_per_parent.csv}"
images_dir="${N2_IMAGES_DIR:-${n2_dir}/images_all}"
output_dir="${N2_OUTPUT_DIR:-${script_dir}/output/clip_full_TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M}"
cache_dir="${N2_CACHE_DIR:-${HOME}/.cache/clip}"
pretrained="${N2_PRETRAINED:-LAION400M}"
gpu="${N2_GPU:-0}"
device="${N2_DEVICE:-cuda}"
precision="${N2_PRECISION:-amp}"
epochs="${N2_EPOCHS:-10}"
batch_size="${N2_BATCH_SIZE:-512}"
workers="${N2_WORKERS:-8}"
learning_rate="${N2_LR:-1e-4}"
weight_decay="${N2_WEIGHT_DECAY:-0.2}"
beta1="${N2_BETA1:-0.9}"
beta2="${N2_BETA2:-0.98}"
epsilon="${N2_EPS:-1e-6}"
warmup_steps="${N2_WARMUP_STEPS:-8000}"
seed="${N2_SEED:-0}"
checkpoint_steps="${N2_CHECKPOINT_STEPS:-500}"
bad_image_policy="${N2_BAD_IMAGE_POLICY:-skip}"

resume_args=()
if [[ -n "${N2_RESUME_CHECKPOINT:-}" ]]; then
    resume_args=(--resume "${N2_RESUME_CHECKPOINT}")
fi

"${N2_PYTHON_BIN:-python}" my_code/train/finetune/train_n2_tinyclip_clip.py \
    --split-csv "${split_csv}" \
    --caption-csv "${caption_csv}" \
    --images-dir "${images_dir}" \
    --output-dir "${output_dir}" \
    --pretrained "${pretrained}" \
    --cache-dir "${cache_dir}" \
    --gpu "${gpu}" \
    --device "${device}" \
    --precision "${precision}" \
    --epochs "${epochs}" \
    --batch-size "${batch_size}" \
    --workers "${workers}" \
    --lr "${learning_rate}" \
    --weight-decay "${weight_decay}" \
    --beta1 "${beta1}" \
    --beta2 "${beta2}" \
    --eps "${epsilon}" \
    --warmup-steps "${warmup_steps}" \
    --seed "${seed}" \
    --checkpoint-steps "${checkpoint_steps}" \
    --bad-image-policy "${bad_image_policy}" \
    "${resume_args[@]}"
