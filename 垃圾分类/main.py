import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别映射
inverted = {
    0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle',
    6: 'Cardboard', 7: 'Basketball', 8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks',
    11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush', 15: 'Dirty Cloth',
    16: 'Seashell', 17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp',
    21: 'Tablet capsules', 22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'
}


# 数据预处理
def image_process(image):
    
    if isinstance(image, Image.Image):
        image = np.array(image)
    # Resize
    image = cv2.resize(image, (224, 224))
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    # HWC -> CHW
    image = image.transpose((2, 0, 1))
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image, dtype=torch.float32).to(device)


# 加载 PyTorch 模型

num_classes = 26
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)
model.eval()

# 加载训练好的权重
checkpoint_path = "./results/py1/best.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
 

# 推理函数

def predict(image_input):
    """
    支持 image_input 为路径字符串 或 numpy.ndarray
    """
    if isinstance(image_input, str):  # 是路径
        image = Image.open(image_input).convert("RGB")
    else:  # 已经是 numpy
        image = image_input
    img_tensor = image_process(image)
    with torch.no_grad():
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1).item()
    return inverted[pred]


# 测试单张图片

image_path = './datasets/5fbdf571c06d3433df85ac65-momodel/garbage_26x100/val/00_01/00037.jpg'
result = predict(image_path)
print("Predicted class:", result)
