# AI 多模态创作引擎

端到端 AI 创作平台，支持文生图、图生图，后续扩展视频生成、风格迁移、3D 重建。

## 功能

- **文生图** — Stable Diffusion 文本到图片生成
- **图生图** — 基于输入图片 + 文本提示修改/增强
- **风格迁移** — Neural Style Transfer，内容图 + 风格图 → 艺术化图片
- **图生视频** — Stable Video Diffusion，单图 → 短视频
- **Web 界面** — Gradio 交互式 Web UI，支持四个功能 Tab

## 项目结构

src/multimodal_engine/ ├── configs/ # 模型配置文件 ├── image_gen/ # 文生图 + 图生图 │ └── pipeline.py ├── style_transfer/ # 风格迁移 (VGG19 + LBFGS) │ ├── vgg.py │ └── transfer.py ├── video_gen/ # 图生视频 (SVD) │ └── pipeline.py └── utils

tests/ # 单元测试 app.py # Gradio Web 入口

## 技术栈

- PyTorch / Diffusers
- Gradio (Web UI)
- YAML 配置驱动
- pytest 单元测试
- Neural Style Transfer (VGG19 + Gram Matrix)

## 快速开始

```bash
pip install -e .
python app.py
# 浏览器打开 http://127.0.0.1:7860
项目结构
src/multimodal_engine/
├── configs/       # 模型配置文件
├── image_gen/     # 图像生成模块
│   ├── pipeline.py    # TextToImage + ImageToImage
│   └── __init__.py
└── utils/         # 工具函数

tests/             # 单元测试
app.py             # Gradio Web 入口
技术栈
PyTorch / Diffusers
Gradio
YAML 配置驱动
pytest 单元测试