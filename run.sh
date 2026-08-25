python run_gptq.py \
  --model_dir models/llama3.1-8b/llama3.1-8b-instruct \
  --save_dir models/llama3.1-8b/llama3.1-8b-gptq-int4-calib \
  --calib_file data/quantization/calibration_wikipedia_multilingual.jsonl 

python main.py \
  --dataset_type klar \
  --checkpoint /root/autodl-tmp/Multilingual_QE/models/llama3.1-8b \
  --quant_type fp16 \
  --model_type llama3.1 \
  --languages ar,ca,el,en,es,fr,he,hu,ja,ko,nl,tr,zh \
  --batch_size 16 \
  --save_path results