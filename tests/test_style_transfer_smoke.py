import torch
from unittest.mock import patch, MagicMock
from multimodal_engine.style_transfer.gatys import StyleTransfer, gram_matrix

SAMPLE_CONFIG = """
model:
  vgg_layers:
    content: ["conv4_2"]
    style: ["conv1_1", "conv2_1"]

optimization:
  steps: 10
  content_weight: 1.0
  style_weight: 1e3
  tv_weight: 0.0
"""


def test_gram_matrix():
    x = torch.randn(1, 3, 4, 4)
    g = gram_matrix(x)
    assert g.shape == (1, 3, 3)


def test_transfer_smoke(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(SAMPLE_CONFIG, encoding="utf-8")

    with patch("multimodal_engine.style_transfer.gatys.VGGFeatureExtractor") as mock_vgg:
        mock_extractor = MagicMock()
        mock_extractor.parameters.return_value = iter([torch.tensor(0.0)])
        mock_vgg.return_value = mock_extractor

        st = StyleTransfer(config_path=str(config_file))

        content = torch.randn(1, 3, 64, 64)
        style = torch.randn(1, 3, 64, 64)

        def mock_forward(x):
            base = x.mean(dim=1, keepdim=True)
            out = {
                "conv4_2": torch.nn.functional.adaptive_avg_pool2d(base, (16, 16)).repeat(1, 128, 1, 1),
            }
            style_out = {
                "conv1_1": torch.nn.functional.adaptive_avg_pool2d(base, (64, 64)).repeat(1, 64, 1, 1),
                "conv2_1": torch.nn.functional.adaptive_avg_pool2d(base, (32, 32)).repeat(1, 128, 1, 1),
            }
            return out, style_out

        mock_extractor.side_effect = mock_forward
        result = st.transfer(content, style)
        assert result.dim() == 3
