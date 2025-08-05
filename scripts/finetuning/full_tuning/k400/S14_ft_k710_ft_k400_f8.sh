
export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
# chmod +x /root/InternVideo/InternVideo2/single_modality/scripts/finetuning/full_tuning/k400/S14_ft_k710_ft_k400_f8.sh
# ./scripts/finetuning/full_tuning/k400/S14_ft_k710_ft_k400_f8.sh
# sed -i 's/\r$//' ./scripts/finetuning/full_tuning/k400/S14_ft_k710_ft_k400_f8.sh
JOB_NAME='S14_ft_k710_ft_k400_f8'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/output_intern_video_format_dataset/videos'
DATA_PATH='/root/.cache/kagglehub/datasets/rohanmallick/kinetics-train-5per/versions/1/kinetics400_5per/kinetics400_5per/output_intern_video_format_dataset'
# MODEL_PATH='/root/InternVideo/InternVideo2/single_modality/model_cache/k400/S14_ft_k710_f8.bin' # k400 ft
# MODEL_PATH='/root/InternVideo/InternVideo2/single_modality/model_cache/k710/S14_ft_k710_f8.bin' # k400 ft
MODEL_PATH='/root/InternVideo/InternVideo2/single_modality/model_cache/distillation/S14_dist_1B_stage2.bin' # k400 ft


GPUS_PER_NODE=2

torchrun --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    run_finetuning.py \
    --model internvideo2_small_patch14_224 \
    --data_path ${DATA_PATH} \
    --eval_data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'Kinetics_sparse' \
    --split ',' \
    --nb_classes 400 \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --steps_per_print 10 \
    --batch_size 16 \
    --update_freq 1 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 8 \
    --num_workers 4 \
    --warmup_epochs 1 \
    --warmup_steps 0 \
    --tubelet_size 1 \
    --epochs 10 \
    --lr 8e-4 \
    --warmup_lr 1e-6 \
    --min_lr 1e-6 \
    --drop_path 0.1 \
    --head_drop_path 0.1 \
    --fc_drop_rate 0.0 \
    --layer_decay 0.75 \
    --layer_scale_init_value 1e-5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --enable_deepspeed \
    --bf16 \
    --zero_stage 1 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --test_best 
    # --eval \
    
