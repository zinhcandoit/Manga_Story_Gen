#!/bin/bash
set -e
uv run python -m scripts.preprocessing
uv run python -m scripts.train