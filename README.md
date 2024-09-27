# Residual Quantization

This repository is built on top of [this repository](https://github.com/RUCAIBox/LC-Rec/tree/main/index).

## Step 1 (Prepare Embeddings)

```bash
python -u generate_indices_distance.py \
  --dataset $Dataset \
  --device cuda:0 \
  --ckpt_path log/$Dataset/llama_256/best_collision_model.pth \
  --output_dir $OUTPUT_DIR \
  --output_file ${Dataset}.index_lemb.json
```

## Step 2 (Export)
```bash
python -u main.py \
  --lr 1e-3 \
  --epochs 10000 \
  --batch_size 1024 \
  --weight_decay 1e-4 \
  --lr_scheduler_type linear \
  --dropout_prob 0.0 \
  --bn False \
  --e_dim 32 \
  --quant_loss_weight 1.0 \
  --beta 0.25 \
  --num_emb_list 256 256 256 256 \
  --sk_epsilons 0.0 0.0 0.0 0.003 \
  --layers 2048 1024 512 256 128 64 \
  --device cuda:2 \
  --data_path ../BenchLLM4RS/export/mind/llama1-embeds.npy \
  --ckpt_dir ./mind.llama1
```