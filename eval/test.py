# test.py
from multimodal_engine.image_gen.pipeline import generate

img = generate("a cat")
img.save("output.png")   # 直接保存，使用相对路径

# 或者使用你给出的动态路径代码：
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "output.png")
img.save(save_path)