# LoRA Training Notes

## Goal

Train a style LoRA for SD 1.5 / AnythingV5 and integrate it into the AIGC engine.

## Dataset

- 195 images
- Deduplication, cropping and tagging
- Resolution: 768px

## Training

- Tool: kohya-ss / sd-scripts
- Base model: SD 1.5 / AnythingV5
- Rank: 64
- Final checkpoint: v5c_e3

## Evaluation

- Hue histogram cosine similarity
- Average saturation
- CFG scale probe
- Group blind review
- ComfyUI re-test

## Result

- Best style similarity: 0.754
- Baseline: 0.734
- Improvement: 2.7%
- Recommended scale: 1.0-1.2

## Note

Model weights and raw dataset are not included due to file size and copyright considerations. Generated samples and evaluation outputs are provided in the portfolio.