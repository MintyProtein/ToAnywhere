accelerate launch train.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" \
    --csv_path="./train.csv" \
    --output_dir="./model_checkpoints/stable_diffusion_inpaint/finetuning" \
    --train_batch_size=1 \
    --num_train_epochs=10 \
