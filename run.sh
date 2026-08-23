python main.py \
  --dataset_type klar \
  --checkpoint /root/autodl-tmp/models/gemma2-9b \
  --quant_type fp16 \
  --model_type gemma2 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 16 \
  --save_path results