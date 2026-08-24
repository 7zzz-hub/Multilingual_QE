python main.py \
  --dataset_type klar \
  --checkpoint /root/autodl-tmp/models/qwen3-8b \
  --quant_type bnb-nf4 \
  --model_type qwen3 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 1 \
  --save_path results