import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Grad-CAMクラスの定義
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 逆伝播時に勾配を取得するフック
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_backward_hook(backward_hook)

        # 順伝播時に特徴マップを取得するフック
        def forward_hook(module, input, output):
            self.activations = output
        self.target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_image, target_class=None):
        output = self.model(input_image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()

        # 勾配と特徴マップを取得
        gradients = self.gradients[0].cpu().data.numpy()  # shape: (C, H, W)
        activations = self.activations[0].cpu().data.numpy()  # shape: (C, H, W)

        # チャネルごとの勾配のグローバル平均値を計算
        weights = np.mean(gradients, axis=(1, 2))  # shape: (C,)

        # 各チャネルの特徴マップに重みを乗じて線形結合
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)  # ReLU適用

        # 入力画像のサイズに合わせてリサイズ（ここでは224x224を仮定）
        cam = cv2.resize(cam, (input_image.size(3), input_image.size(2)))
        
        # CAMを0-1に正規化
        cam -= np.min(cam)
        if np.max(cam) != 0:
            cam /= np.max(cam)
        return cam

# --------------------------------------------------
# ファインチューニング済みモデルのロード例
# --------------------------------------------------
# 例：VGG16をベースにファインチューニングしている場合
model = vgg16(weights=VGG16_Weights.DEFAULT)
# 例として分類クラス数が10の場合（学習時の設定に合わせる）
num_classes = 4
model.classifier[6] = torch.nn.Linear(4096, num_classes)

# ファインチューニング済みモデルの重みをロード
model_path = "thai/best_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Grad-CAMの対象層を指定（通常は最後の畳み込み層）
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

# --------------------------------------------------
# 画像の前処理
# --------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 画像の読み込み（適宜パスを変更してください）
img_path = "thai/mydesk_raw/run3/20250228T165549_0011.jpg"
img = Image.open(img_path).convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)  # バッチ次元を追加

# Grad-CAMでヒートマップを生成
cam = grad_cam.generate_cam(input_tensor)

# --------------------------------------------------
# ヒートマップと元画像の可視化および保存
# --------------------------------------------------
# 元画像をnumpy配列に変換（リサイズ済み）
img_np = np.array(img.resize((224, 224)))

# ヒートマップをJETカラーマップに変換
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
# OpenCVはBGRなので、RGBに変換
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# ヒートマップと元画像を重ね合わせ
overlay = heatmap * 0.4 + img_np * 0.6

# プロットで確認
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(img_np)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(heatmap)
plt.title("Grad-CAM Heatmap")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.uint8(overlay))
plt.title("Overlay")
plt.axis('off')

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 画像として保存
# --------------------------------------------------
# cv2.imwriteはBGR形式を想定しているため、RGBからBGRに変換して保存
heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
overlay_bgr = cv2.cvtColor(np.uint8(overlay), cv2.COLOR_RGB2BGR)

cv2.imwrite("heatmap.jpg", heatmap_bgr)
cv2.imwrite("overlay.jpg", overlay_bgr)

print("HeatmapとOverlayがそれぞれ 'heatmap.jpg' と 'overlay.jpg' として保存されました。")
