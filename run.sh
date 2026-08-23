python main.py \
  --dataset_type klar \
  --checkpoint /root/autodl-tmp/models/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 \
  --quant_type gptq-int4 \
  --model_type llama \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 16 \
  --save_path results