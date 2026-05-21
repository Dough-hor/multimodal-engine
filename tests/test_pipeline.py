import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("USERNAME", "default_user")
import torch
import pytest
from unittest.mock import patch, MagicMock
from multimodal_engine.image_gen.pipeline import TextToImage
SAMPLE_CONFIG = """
model:
  model_id: "test/model"
  device: "cpu"

inference:
  steps: 30
  guidance_scale: 8.0
  negative_prompt: "blurry, ugly, lowres, bad anatomy"
  width: 512
  height: 512
  seed: 42
"""

SAMPLE_IMG2IMG_CONFIG = """
model:
  model_id: "test/model"
  device: "cpu"

inference:
  steps: 20
  guidance_scale: 5.0
  strength: 0.6
  negative_prompt: "bad quality"
  seed: 123
"""


@pytest.fixture
def temp_config_file(tmp_path):
    """创建一个临时配置文件，返回路径"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(SAMPLE_CONFIG,encoding="utf-8")
    return str(config_file)

@pytest.fixture
def temp_img2img_config_file(tmp_path):
    config_file = tmp_path / "img2img_config.yaml"
    config_file.write_text(SAMPLE_IMG2IMG_CONFIG, encoding="utf-8")
    return str(config_file)

def test_config_loading(temp_config_file):
    with patch("multimodal_engine.image_gen.pipeline.StableDiffusionPipeline.from_pretrained") as mock_from_pretrained:
      # 创建一个假的 pipeline 实例，并让 .to 返回它自己
      mock_pipe=MagicMock()
      mock_from_pretrained.return_value=mock_pipe
      mock_pipe.to.return_value=mock_pipe

      # 实例化 TextToImage，传入临时配置路径

      t2i = TextToImage(config_path=temp_config_file)

      # 验证模型加载方法被调用了正确的参数（但实际未执行）
      mock_from_pretrained.assert_called_once_with("test/model", torch_dtype=torch.float32)
      mock_pipe.to.assert_called_once_with("cpu")

      # 验证配置被正确读取到实例属性
      assert t2i.model_id == "test/model"
      assert t2i.device == "cpu"
      assert t2i.default_steps == 30
      assert t2i.default_cfg_scale == 8.0
      assert t2i.default_negative_prompt == "blurry, ugly, lowres, bad anatomy"
      assert t2i.default_width == 512
      assert t2i.default_height == 512
      assert t2i.default_seed == 42

def test_img2img_config_loading(temp_img2img_config_file):
    from multimodal_engine.image_gen.pipeline import ImageToImage
    with patch("multimodal_engine.image_gen.pipeline.StableDiffusionImg2ImgPipeline.from_pretrained") as mock_from_pretrained:
        mock_pipe=MagicMock()
        mock_from_pretrained.return_value=mock_pipe
        mock_pipe.to.return_value=mock_pipe
        
        i2i = ImageToImage(config_path=temp_img2img_config_file)

        assert i2i.default_steps == 20
        assert i2i.default_cfg_scale == 5.0
        assert i2i.default_strength == 0.6
        assert i2i.default_negative_prompt == "bad quality"
        assert i2i.default_seed == 123


def test_missing_config():
    with pytest.raises(FileNotFoundError):
        TextToImage(config_path="/non/existent/path/config.yaml")