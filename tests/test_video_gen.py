import pytest
from unittest.mock import patch, MagicMock
from multimodal_engine.video_gen.pipeline import ImageToVideo

# 模拟的配置文件内容（与 video_config.yaml 结构一致）
SAMPLE_CONFIG = """
model:
  model_id: "test/video-model"
  device: "cpu"

inference:
  frames: 8
  fps: 4
  decode_chunk_size: 4
  motion_bucket_id: 100
  noise_aug_strength: 0.05
  seed: 42
"""

@pytest.fixture
def temp_config_file(tmp_path):
    """创建一个临时配置文件，返回路径"""
    config_file = tmp_path / "video_config.yaml"
    config_file.write_text(SAMPLE_CONFIG, encoding="utf-8")
    return str(config_file)


def test_config_loading(temp_config_file):
    """测试配置加载：能正确读取 model_id、device、frames、fps 等参数"""
    with patch("multimodal_engine.video_gen.pipeline.StableVideoDiffusionPipeline.from_pretrained") as mock_from_pretrained:
        mock_pipe = MagicMock()
        mock_from_pretrained.return_value = mock_pipe

        # 实例化 ImageToVideo
        vgen = ImageToVideo(config_path=temp_config_file)

        # 验证模型被正确加载
        mock_from_pretrained.assert_called_once()
        assert vgen.model_id == "test/video-model"
        assert vgen.device == "cpu"
        assert vgen.default_frames == 8
        assert vgen.default_fps == 4
        assert vgen.default_chunk_size == 4
        assert vgen.default_motion == 100
        assert vgen.default_noise_aug == 0.05
        assert vgen.default_seed == 42


def test_missing_config():
    """配置文件不存在时应抛出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        ImageToVideo(config_path="/non/existent/path/config.yaml")
