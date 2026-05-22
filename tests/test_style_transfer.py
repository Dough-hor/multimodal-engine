import pytest
from unittest.mock import patch, MagicMock
from multimodal_engine.style_transfer.transfer import StyleTransfer

# 模拟的配置文件内容（与 style_transfer_config.yaml 结构一致）
SAMPLE_CONFIG = """
model:
  vgg_layers:
    content: ["conv4_2"]
    style: ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

optimization:
  steps: 300
  content_weight: 1.0
  style_weight: 1e6
  tv_weight: 0.0
"""

@pytest.fixture
def temp_config_file(tmp_path):
    """创建一个临时配置文件，返回路径"""
    config_file = tmp_path / "style_transfer_config.yaml"
    config_file.write_text(SAMPLE_CONFIG, encoding="utf-8")
    return str(config_file)


def test_config_loading(temp_config_file):
    """测试配置加载：能正确读取 content_layers, style_layers 及优化参数"""
    # Mock 掉 VGGFeatureExtractor，避免真正加载模型
    with patch("multimodal_engine.style_transfer.transfer.VGGFeatureExtractor") as mock_vgg:
        mock_extractor = MagicMock()
        mock_vgg.return_value = mock_extractor

        # 实例化 StyleTransfer
        st = StyleTransfer(config_path=temp_config_file)

        # 验证 VGGFeatureExtractor 被正确调用（参数来自配置文件）
        mock_vgg.assert_called_once_with(
            ["conv4_2"],
            ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        )

        # 验证配置读取正确
        assert st.content_layers == ["conv4_2"]
        assert st.style_layers == ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        assert st.steps == 300
        assert st.content_weight == 1.0
        assert st.style_weight == 1e6
        assert st.tv_weight == 0.0


def test_missing_config():
    """配置文件不存在时应抛出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        StyleTransfer(config_path="/non/existent/path/config.yaml")