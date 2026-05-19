import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from multimodal_engine.image_gen.pipeline import generate

img = generate("a cute cat standing on a table, cartoon style")
img.save("test_output.png")
print("图片已保存为 test_output.png")