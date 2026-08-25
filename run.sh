  # --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \

python main.py \
  --dataset_type klar \
  --checkpoint /root/autodl-tmp/Multilingual_QE/models/llama3.1-8b-GPTQ-INT4 \
  --quant_type gptq-int4-multilingualism \
  --model_type llama3.1 \
  --languages ar,ca,en,zh \
  --batch_size 16 \
  --save_path results

python main.py \
  --dataset_type klar \
  --checkpoint /root/autodl-tmp/Multilingual_QE/models/llama3.1-8b-gptq-int4 \
  --quant_type gptq-int4 \
  --model_type llama3.1 \
  --languages ar,ca,en,zh \
  --batch_size 16 \
  --save_path results

# python main.py \
#   --dataset_type klar \
#   --checkpoint /root/autodl-tmp/Multilingual_QE/models/llama3.1-8b \
#   --quant_type fp16 \
#   --model_type llama3.1 \
#   --languages ar,ca,en,zh \
#   --batch_size 16 \
#   --save_path results