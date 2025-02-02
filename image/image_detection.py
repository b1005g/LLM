from PIL import Image
from transformers import MllamaForConditionalGeneration,MllamaProcessor
from transformers import BitsAndBytesConfig
import torch
import glob
import os

# 모델 먼저 불러오기
def load_model():
    save_directory = "./llama-3.2-Korean-Bllossom-AICA-5B_Q4"
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,                       
        llm_int8_enable_fp32_cpu_offload=True,   
    )
    model = MllamaForConditionalGeneration.from_pretrained(
        save_directory,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    processor = MllamaProcessor.from_pretrained(save_directory)
    return model, processor

def load_image(image_dir):
    image_files = glob.glob(os.path.join(image_dir, "*.*"))
    image_path = image_files[0]
    image = Image.open(f'{image_path}').convert('RGB')
    return image

def image_detect(model, processor, image):
    messages = [
    {'role': 'user','content': [
        {'type':'image'},
        {'type': 'text','text': '이 이미지에 대해서 설명해줘'}
        ]},
    ]
    input_text = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
    )

    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    output = model.generate(**inputs, max_new_tokens=256,temperature=0.1,eos_token_id=processor.tokenizer.convert_tokens_to_ids('<|eot_id|>'),use_cache=False)
    result = processor.decode(output[0])
    return result

#os.remove(image_path)