import os
import random
import logging
import numpy as np
import shutil
import tomllib
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------
# 再現性のためのシード固定関数
def set_seed(seed=57):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(57)

# -------------------------
# config.toml を読み込む
with open("config.toml", "rb") as f:  # `rb` モードで開く必要がある
    config = tomllib.load(f)

num_classes = config["hyperparameters"]["num_classes"]
batch_size = config["hyperparameters"]["batch_size"]
epochs = config["hyperparameters"]["epochs"]
img_size = config["hyperparameters"]["img_size"]
learning_rate = config["hyperparameters"]["learning_rate"]
data_dir = config["directory"]["data_dir"]
result_dir = config["directory"]["result_dir"]
gpu = config["gpu"]["gpu_index"]

# -------------------------
# 時間を記録
now = datetime.now()
formatted_time = now.strftime("%Y%m%dT%H%M%S")
result_dir = os.path.join(result_dir, formatted_time)

# -------------------------
# ログファイル作成
os.makedirs(f"{result_dir}", exist_ok=True)
shutil.copy("config.toml", f"{result_dir}/config.toml")
log_file = f"{result_dir}/log.txt"

# ログ設定
logging.basicConfig(
    filename=log_file,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logging.info("訓練開始: {}".format(formatted_time))
logging.info(f"データディレクトリ: {data_dir}")
logging.info(f"結果保存ディレクトリ: {result_dir}")
logging.info(f"使用GPU: {gpu}")

# -------------------------
# データ前処理
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# データセットとデータローダー
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, transform=data_transforms['val'])
}

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
}

logging.info("データセットのロード完了")
logging.info(f"学習データ: {len(image_datasets['train'])}枚")
logging.info(f"検証データ: {len(image_datasets['val'])}枚")

# -------------------------
# デバイス設定
device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")
logging.info(f"使用デバイス: {device}")

# -------------------------
# モデル構築
model = vgg16(weights=VGG16_Weights.DEFAULT)

# 特徴抽出部分を固定
for param in model.features.parameters():
    param.requires_grad = False

# 最終層をクラス数に合わせて変更
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)

logging.info(f"モデル: VGG16（最終層の出力を {num_classes} クラスに変更）")

# -------------------------
# 損失関数と最適化
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# -------------------------
# 訓練と検証ループ
best_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

logging.info("学習開始")

for epoch in range(epochs):
    logging.info(f"Epoch {epoch+1}/{epochs} 開始")
    print(f'Epoch {epoch+1}/{epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
        
        running_loss = 0.0
        running_corrects = 0

        dataloader = dataloaders[phase]
        progress_bar = tqdm(dataloader, desc=phase)
        
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.item())
        else:
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc.item())

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), f'{result_dir}/vgg_best_model.pth')
                logging.info(f"新しいベストモデルを保存: Acc {best_acc:.4f}")
                print("Best model saved!")

    scheduler.step()
    print()

# -------------------------
# 学習曲線のプロット
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accs, label='Train Acc')
plt.plot(range(1, epochs+1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(f'{result_dir}/loss_accuracy_plot.png', dpi=300, bbox_inches='tight')
logging.info("学習曲線のプロットを保存")

plt.show()

# -------------------------
# 最終モデルの保存
torch.save(model.state_dict(), f'{result_dir}/vgg_final_model.pth')
logging.info("最終モデルを保存")

logging.info("学習完了")
