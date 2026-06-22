# Manga Story Generation

## 1. Input & Output
- Input: `prompt` + `N images` (N > 0)
- Output: `<think>` + generated story in `<story>`

## 2. Environment Preparation
```bash
uv sync
```

## 3. Training
- **Prepare dataset**:

Run in terminal:
```bash
uv run python -m scripts.preprocessing
```
- **Run training pipeline**:

a) Tuning parameters in `configs/train.yaml`

b) Run in terminal:
```bash
uv run python -m scripts.train --target_size 3500 --max_pages 15
# Or
sh train.sh
```
## Note
- This is dataset link: https://huggingface.co/datasets/TQZinh/Manga_StoryGen
- Trained model: https://huggingface.co/TQZinh/Manga_Story_Gen_SFT
