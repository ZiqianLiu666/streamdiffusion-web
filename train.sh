#!/bin/sh

#SBATCH --job-name=time_0   # Job name
#SBATCH --nodes=1
#SBATCH --partition=H100,A100
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=60GB
#SBATCH --time=24:00:00
#SBATCH --output=reports/train_output.txt
#SBATCH --error=reports/train_error.txt

# Run training script - optimized to complete within one day
# Key optimizations:
# 1. validation_steps=2000: Changed from per-step validation to every 2000 steps (saves 99% validation time)
# 2. max_train_steps=40000: Reduced from 450000 steps to 40000 steps
# 3. gradient_accumulation_steps=4: Simulates batch_size=4 without increasing memory

python SwiftEdit/train_ip_s2_ldist_lnoise_v3.py \
  --data_option                stream_tsv \
  --task                       reconstruct \
  --pretrained_model_name_or_path  Manojb/stable-diffusion-2-1-base \
  --pretrained_teacher_denoise     Manojb/stable-diffusion-2-1-base \
  --pretrained_sb_generator        stabilityai/sd-turbo \
  --pretrained_ip_s1_path          SwiftEdit/swiftedit_weights \
  --image_encoder_path             h94/IP-Adapter \
  --output_dir                     outputs/sd_turbo_stage2 \
  --logging_dir                    logs \
  --learning_rate                  1e-5 \
  --train_batch_size               1 \
  --gradient_accumulation_steps    4 \
  --validation_steps               2000 \
  --max_train_steps                50000 \
  --checkpointing_steps            2000 \
  --use_ema

