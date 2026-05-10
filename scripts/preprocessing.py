import os
import re
import json
import math
import random
import zipfile
from huggingface_hub import snapshot_download
from datasets import Dataset
random.seed(42)

# 1. Cấu hình thông tin
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(ROOT_DIR, "data")
REPO_ID = "TQZinh/Manga_Story_Gen"
MANGA_DIR = os.path.join(LOCAL_DIR, 'manga_dataset/image')
FINAL_TRAIN_DIR = os.path.join(LOCAL_DIR, 'manga_dataset/final/train')
COT_DIR = os.path.join(LOCAL_DIR, 'manga_dataset/synthetic_cot')
STORY_DIR = os.path.join(LOCAL_DIR, 'manga_dataset/generated_story')
GENRE_PATH = os.path.join(LOCAL_DIR, 'manga_detail.json')

existing_files = os.listdir(LOCAL_DIR)
if "manga_dataset" not in existing_files or "manga_detail.json" not in existing_files:
    print(f"--- Đang bắt đầu tải dataset từ {REPO_ID} ---")

    # 2. Tải toàn bộ repository về
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        token=os.environ.get("HF_TOKEN"), 
        local_dir_use_symlinks=False
    )

    print(f"✅ Đã tải xong về: {path}")

    # 3. Tự động tìm và giải nén các file .zip
    print("--- Đang kiểm tra và unzip các file nén ---")
    for file in os.listdir(LOCAL_DIR):
        if file.endswith(".zip"):
            zip_path = os.path.join(LOCAL_DIR, file)
            print(f"📦 Đang giải nén: {file}...")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Giải nén ngay tại thư mục hiện tại
                    zip_ref.extractall(LOCAL_DIR)
                print(f"   ➤ Thành công!")
                os.remove(zip_path)
            except Exception as e:
                print(f"   ❌ Lỗi khi giải nén {file}: {e}")

    print("--- HOÀN TẤT ---")
else: print("Downloaded data!")

# Training Helper
def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def load_genre_data():
    if not os.path.exists(GENRE_PATH):
        return {}
    with open(GENRE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    result = {}
    for item in data:
        genres = item.get('genres', [])
        result[item['name']] = genres
        normalized = re.sub(r'\s*\(?:WN|LN\)\s*$', '', item['name']).strip()
        result[normalized] = genres
    return result

CLUSTER_RULES = [
    ("dark_mature",      {"Horror", "Tragedy", "Psychological", "Mature", "Adult"}),
    ("mystery_thriller", {"Mystery", "Supernatural"}),
    ("romance_drama",    {"Romance", "Drama", "Shoujo", "Josei"}),
    ("school_slice",     {"School Life", "Slice of Life"}),
    ("ecchi_harem",      {"Ecchi", "Harem"}),
    ("mecha_scifi",      {"Mecha", "Sci-fi"}),
    ("comedy_light",     {"Comedy"}),
    ("shounen_battle",   {"Shounen"}),
    ("isekai_adventure", {"Adventure", "Fantasy"}),
]

def assign_cluster(title, genre_data_map):
    genres = set(genre_data_map.get(title, []))
    for cluster_name, trigger_genres in CLUSTER_RULES:
        if genres & trigger_genres:
            return cluster_name
    return "default"

def resolve_image_paths(json_data):
    title = json_data['name']
    chap = str(json_data['chap'])
    start_file = json_data['start']
    num_pages = int(json_data['num_pages'])
    
    manga_title_dir = os.path.join(MANGA_DIR, title)
    if not os.path.isdir(manga_title_dir): return []
    
    chap_dir = os.path.join(manga_title_dir, f"Chap_{chap}")
    if not os.path.isdir(chap_dir):
        possible_dirs = []
        for d in os.listdir(manga_title_dir):
            pot_dir = os.path.join(manga_title_dir, d)
            if os.path.isdir(pot_dir) and start_file in os.listdir(pot_dir):
                possible_dirs.append((pot_dir, d))
        found = False
        if len(possible_dirs) == 1:
            chap_dir = possible_dirs[0][0]
            found = True
        elif len(possible_dirs) > 1:
            for pot_dir, d_name in possible_dirs:
                if re.search(rf'(?<!\d){chap}(?!\d)', d_name):
                    chap_dir = pot_dir
                    found = True
                    break
            if not found: chap_dir, found = possible_dirs[0][0], True
        if not found: return []
    
    all_images = sorted(
        [f for f in os.listdir(chap_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))],
        key=natural_sort_key
    )
    if start_file not in all_images: return []
    start_idx = all_images.index(start_file)
    selected = all_images[start_idx : start_idx + num_pages]
    return [os.path.join(chap_dir, f) for f in selected]

def prepare_manga_data(sampled_files, genre_data_map):
    """
    Hàm duyệt qua các file JSON, load nội dung CoT/Story và build cấu hình prompt cho GRPO.
    """
    data_list = []
    
    # Định nghĩa Prompt cố định
    system_prompt = """You are a professional novel writer. Your task is to write the novel chapter inside <story>...</story> and reasoning inside <think>...</think>.
You NEVER output anything outside these tag pairs. You NEVER skip or omit <story>...</story> tags or <think>...</think> tag."""
    user_instruction = """Let's think step by step. Analyze the provided manga panels step-by-step inside <think> tags.
Then write a compelling novel chapter inside <story> tags based on the visuals.
REMINDER: You MUST wrap your story in <story>...</story>."""
    
    for fname in sampled_files:
        json_path = os.path.join(FINAL_TRAIN_DIR, fname)
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        cluster = assign_cluster(json_data['name'], genre_data_map)
        image_paths = resolve_image_paths(json_data)
        
        if not image_paths:
            continue
        
        # Load CoT
        cot_path = os.path.join(COT_DIR, cluster, fname.replace('.json', '_cot.md'))
        if not os.path.exists(cot_path): continue
        with open(cot_path, 'r', encoding='utf-8') as f:
            cot_content = f.read().strip()
        if not cot_content: continue
        
        # Load Story
        story_path = os.path.join(STORY_DIR, cluster, fname.replace('.json', '_story.json'))
        if not os.path.exists(story_path): continue
        with open(story_path, 'r', encoding='utf-8') as f:
            story_content = json.load(f).get("story", "")
        if not story_content.strip(): continue
        
        # ==========================================
        # ĐOẠN ĐÃ SỬA: Xây dựng cấu trúc prompt Multi-role
        # ==========================================
        prompt = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": []} # Khởi tạo content rỗng cho user
        ]
        
        # Bước 1: Thêm TẤT CẢ các node image vào TRƯỚC
        for _ in image_paths:
            prompt[1]["content"].append({"type": "image"})
            
        # Bước 2: Chốt lại bằng câu lệnh Text ở CUỐI CÙNG
        prompt[1]["content"].append({"type": "text", "text": user_instruction})
        # ==========================================
                
        full_answer = f"<think>\n{cot_content}\n</think>\n<story>\n{story_content.strip()}\n</story>"
        
        data_list.append({
            "prompt": prompt,
            "image_paths": image_paths,
            "num_pages": len(image_paths),
            "answer": full_answer,
            "title": json_data['name'],   
            "chap": str(json_data['chap']), 
            "cluster": cluster
        })
        
    return data_list
    
def get_gaussian_sampled_dataset(data_list, target_size=-1, max_pages = -1):
    """
    Hàm thực hiện lọc và lấy mẫu dữ liệu theo phân phối chuẩn (Gaussian).
    """
    # 1. Lọc theo số trang tối đa
    if max_pages != -1:
        filtered_data = [d for d in data_list if d['num_pages'] <= max_pages]
        if not filtered_data:
            raise ValueError(f"Không có dữ liệu thỏa mãn num_pages <= {max_pages}.")
    else:
        filtered_data = data_list
        max_pages = max(d['num_pages'] for d in filtered_data)

    if target_size == -1 or target_size >= len(filtered_data):
        random.shuffle(filtered_data)
        return Dataset.from_list(filtered_data)

    # 2. Tính toán tham số phân phối chuẩn
    min_pages = min(d['num_pages'] for d in filtered_data)
    mu = (min_pages + max_pages) / 2.0
    sigma = (max_pages - min_pages) / 4.0
    
    # 3. Tính tỷ lệ lý tưởng (Normal Ratios)
    normal_weights = {}
    total_weight = 0
    for pages in range(min_pages, max_pages + 1):
        weight = math.exp(-0.5 * ((pages - mu) / sigma) ** 2)
        normal_weights[pages] = weight
        total_weight += weight
    
    normal_ratios = {k: v / total_weight for k, v in normal_weights.items()}
    
    # 4. Lấy mẫu theo quota
    sampled_data = []
    for pages, ratio in normal_ratios.items():
        quota = int(round(target_size * ratio))
        candidates = [d for d in filtered_data if d['num_pages'] == pages]
        
        if quota >= len(candidates):
            sampled_data.extend(candidates)
        else:
            sampled_data.extend(random.sample(candidates, quota))
            
    # 5. Hậu xử lý (Giới hạn tối đa và trộn ngẫu nhiên)
    if len(sampled_data) > target_size:
        sampled_data = random.sample(sampled_data, target_size)
    
    random.shuffle(sampled_data)
    
    print(f"✅ Đã lọc còn {len(filtered_data)} mẫu (num_pages <= {max_pages}).")
    print(f"✅ Đã sample thành công {len(sampled_data)} mẫu (Gaussian Shape).")
    
    return Dataset.from_list(sampled_data)