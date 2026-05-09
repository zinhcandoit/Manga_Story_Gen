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
uv run python -m scripts.train
# Or
sh train.sh
```