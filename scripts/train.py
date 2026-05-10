import os
import re
import gc
import json
import random
import torch
torch._dynamo.config.disable = True
import torch.nn.functional as F
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import matplotlib.pyplot as plt
import math
from collections import Counter
from datasets import Dataset
from transformers import AutoProcessor, TextStreamer
from sentence_transformers import SentenceTransformer
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastVisionModel, is_bfloat16_supported
import yaml
import argparse
from scripts.preprocessing import (
    load_genre_data, 
    prepare_manga_data,
    get_gaussian_sampled_dataset,
    FINAL_TRAIN_DIR
)
def main():
    # 1. Cấu hình Argument Parser để nhận tham số từ terminal
    parser = argparse.ArgumentParser(description="Manga Training Pipeline")
    parser.add_argument("--target_size", type=int, default=-1, help="Samples")
    parser.add_argument("--max_pages", type=int, default=-1, help="Max Images for each sample")
    cli_args = parser.parse_args()
    with open("config/train.yaml", "r") as f:
        config = yaml.safe_load(f)
    genre_data_map = load_genre_data()
    all_train_files = [f for f in os.listdir(FINAL_TRAIN_DIR) if f.endswith('.json')]
    data_list = prepare_manga_data(all_train_files, genre_data_map)

    dataset = get_gaussian_sampled_dataset(
        data_list, 
        target_size=cli_args.target_size, 
        max_pages=cli_args.max_pages
    )

    def transform_fn(examples):
        images = []
        MAX_PIXELS = 360 * 480  # Giới hạn tổng số pixel (tương đương ảnh 512x512)
        
        for paths in examples["image_paths"]:
            batch_images = []
            for p in paths:
                img = Image.open(p).convert("RGB")
                w, h = img.size
                
                # Tính toán tỷ lệ nếu vượt quá giới hạn
                if w * h > MAX_PIXELS:
                    scale = math.sqrt(MAX_PIXELS / (w * h))
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                
                batch_images.append(img)
            images.append(batch_images)
            
        examples["images"] = images
        return examples
    dataset.set_transform(transform_fn)
    print(f"Prepared Dataset for training")

    def extract_solution(generation: str):
        # Dùng regex để lấy nội dung giữa <story> và </story>
        matches = re.findall(r'<story>(.*?)</story>', generation, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            # Lấy khối story cuối cùng có nội dung đáng kể
            for m in reversed(matches):
                if len(m.strip()) > 10:
                    return m.strip()
            return matches[-1].strip()
        return "" 
    def format_reward_func(completions, **kwargs):
        """Thưởng cho việc tuân thủ định dạng và PHẠT NẶNG nếu bị cắt cụt"""
        rewards = []
        for completion in completions:
            text = completion[0]['content'] if isinstance(completion, list) else str(completion)
            reward = 0.0
            
            # Thưởng nếu có đủ cặp thẻ
            if "<think>" in text and "</think>" in text:
                reward += 0.2
            if "<story>" in text and "</story>" in text:
                reward += 0.3
            # Thưởng nếu thẻ <story> nằm sau thẻ </think>
            if "</think>" in text and "<story>" in text:
                if text.find("</think>") < text.find("<story>"):
                    reward += 0.2
                    
            # --- PHẦN HÌNH PHẠT MỚI (CHỐNG LỖI LOSS = 0) ---
            if "</story>" not in text:
                reward -= 1.0  # Phạt nặng nhất: Chưa viết xong đã hết token
            elif not text.strip().endswith("</story>"):
                reward -= 0.5  # Phạt nhẹ: Đã đóng thẻ nhưng còn lảm nhảm thêm rác ở sau
            # -----------------------------------------------
            
            rewards.append(reward)
        return rewards
        
    embed_model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    embed_model.eval()
    def cosine_similarity_reward(prompts, completions, answer, **kwargs):
        all_completion_texts = []
        for completion_node in completions:
            # Unpack text từ format Chat
            if isinstance(completion_node, list) and len(completion_node) > 0 and 'content' in completion_node[0]:
                all_completion_texts.append(completion_node[0]['content'])
            else:
                all_completion_texts.append(str(completion_node))
                
        # 1. Trích xuất story và kiểm tra lỗi định dạng cho cả batch
        gen_solutions = []
        penalties = []
        for text in all_completion_texts:
            sol = extract_solution(text)
            gen_solutions.append(sol)
            
            # Hình phạt nếu thiếu các thẻ quan trọng
            penalty = 0.0
            if "<think>" not in text or "</think>" not in text:
                penalty += 0.25
            if "<story>" not in text or "</story>" not in text:
                penalty += 0.25
            penalties.append(penalty)
            
        # 2. Batch Encoding (Nhanh hơn rất nhiều so với chạy vòng lặp)
        with torch.no_grad():
            # encode hỗ trợ truyền vào một list string
            embeddings_gen = embed_model.encode(gen_solutions, convert_to_tensor=True, device='cuda')
            ref_solutions = [extract_solution(ans) if '<story>' in ans else ans for ans in answer]
            embeddings_ref = embed_model.encode(ref_solutions, convert_to_tensor=True, device='cuda')
            
            # Tính similarity cho cả batch một lúc
            sim_scores = F.cosine_similarity(embeddings_gen, embeddings_ref)
            
        # 3. Tổng hợp reward và trừ penalty
        # sim_scores ở dạng tensor, ta trừ đi tensor penalties
        rewards = (sim_scores - torch.tensor(penalties, device=sim_scores.device)).tolist()
        
        return rewards

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name     = config["model_name"],
        max_seq_length = config["max_seq_length"],
        load_in_4bit   = config["load_in_4bit"],
        fast_inference = False,
        trust_remote_code=True
    )
    
    # 3. Cấu hình LoRA (PEFT)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False, 
        finetune_language_layers   = True,  
        finetune_attention_modules = True,  
        finetune_mlp_modules       = True,  
        r                          = config["lora_r"],           
        lora_alpha                 = config["lora_alpha"],  
        lora_dropout               = config["lora_dropout"],
        bias                       = config["bias"],
        random_state               = config.get("seed", 3407),
        use_gradient_checkpointing = "unsloth"
    )

    # Switch model back to training mode
    FastVisionModel.for_training(model)
    model_specific_keys = ["model_name", "max_seq_length", "load_in_4bit", "lora_r", "lora_alpha", "lora_dropout", "bias", "seed"]
    grpo_kwargs = {k: v for k, v in config.items() if k not in model_specific_keys}


    training_args = GRPOConfig(
        **grpo_kwargs,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            format_reward_func,
            cosine_similarity_reward   # Hàm reward bạn đã định nghĩa
            # Bạn có thể thêm các hàm reward định dạng ở đây nếu cần
        ],
        args=training_args,
        train_dataset=dataset,
    )

    torch.cuda.empty_cache()
    gc.collect()
    trainer.train()
    trainer.push_to_hub(config["hub_model_id"])

if __name__ == "__main__":
    main()