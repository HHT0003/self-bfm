#!/usr/bin/env bash
# Offline G1 encoder + FSQ + g1_kin reconstruction on mixed G1 NPZ sources.
# Launches with nohup and logs curves to wandb project general_self_bfm.
# Does not launch Isaac Lab / PPO / g1_dyn.

set -euo pipefail

REPO_ROOT="/data/init_precise_control/projects/GR00T-WholeBodyControl"
OUT_DIR="${REPO_ROOT}/logs/g1_kin_offline"
SCRIPT="${REPO_ROOT}/gear_sonic/scripts/train_g1_kin_offline.py"
RUN_NAME="${RUN_NAME:-g1_kin_offline_s10_fs5_mlp3_mixed}"
GPU_ID="${GPU_ID:-0}"
LOG_FILE="${OUT_DIR}/nohup_${RUN_NAME}.log"

ISAAC_PYTHON="${ISAAC_PYTHON:-/isaac-sim/python.sh}"

DATA_DIRS=(
    "/data/init_precise_control/motion_data/BONES-SEED/g1_mimic_npz_50hz_prefix1of3/all"
    "/data/init_precise_control/motion_data/perceptive_generalist_data_upload/box_omnicontact_contact_augmented_0722_forcevalid_20260727_053934"
    "/data/init_precise_control/motion_data/perceptive_generalist_data_upload/kick"
    "/data/init_precise_control/motion_data/perceptive_generalist_data_upload/lafan1_g1"
    "/data/init_precise_control/motion_data/AMASS_Retargeted_for_G1/g1_mimic_npz_50hz"
)

cd "${REPO_ROOT}"

for data_dir in "${DATA_DIRS[@]}"; do
    if [[ ! -d "${data_dir}" ]]; then
        echo "[ERROR] data dir does not exist: ${data_dir}" >&2
        exit 1
    fi
done
if [[ ! -x "${ISAAC_PYTHON}" ]]; then
    echo "[ERROR] Isaac python does not exist: ${ISAAC_PYTHON}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

echo "[INFO] python=${ISAAC_PYTHON}"
echo "[INFO] data_dirs:"
printf '  %s\n' "${DATA_DIRS[@]}"
echo "[INFO] out_dir=${OUT_DIR}"
echo "[INFO] wandb project=general_self_bfm run=${RUN_NAME}"
echo "[INFO] GPU_ID=${GPU_ID} log=${LOG_FILE}"
echo "[INFO] Encoder/decoder use three layers 2048,1024,512 (not SONIC's four-layer 2048,1024,512,512)."

nohup env CUDA_VISIBLE_DEVICES="${GPU_ID}" WANDB_MODE=online PYTHONUNBUFFERED=1 \
    "${ISAAC_PYTHON}" "${SCRIPT}" train \
    --data_dir "${DATA_DIRS[@]}" \
    --out_dir "${OUT_DIR}" \
    --seq_len 10 \
    --frame_stride 5 \
    --window_stride 5 \
    --enc_hidden_dims 2048,1024,512 \
    --dec_hidden_dims 2048,1024,512 \
    --max_num_tokens 2 \
    --token_dim 32 \
    --fsq_levels 32 \
    --batch_size 256 \
    --epochs 250 \
    --lr 2e-4 \
    --num_workers 8 \
    --device cuda \
    --val_ratio 0.05 \
    --seed 42 \
    --use_wandb \
    --wandb_project general_self_bfm \
    --wandb_run_name "${RUN_NAME}" \
    --wandb_tags g1_kin_offline,fsq,bones_prefix1of3,box,kick,lafan1,amass \
    --wandb_mode online \
    --wandb_log_every_steps 20 \
    "$@" \
    > "${LOG_FILE}" 2>&1 &
TRAIN_PID=$!

echo "[INFO] launched PID=${TRAIN_PID}"
echo "[INFO] log=${LOG_FILE}"
echo "[INFO] wandb project=general_self_bfm"
echo "[INFO] follow with: tail -f ${LOG_FILE}"
