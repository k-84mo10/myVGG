import os
import random
import numpy as np
import toml
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
config = toml.load("config.toml")

num_classes = config["hyperparameters"]["num_classes"]        # クラス数（例: 2クラス分類）
batch_size = config["hyperparameters"]["batch_size"]
epochs = config["hyperparameters"]["epochs"]
img_size = config["hyperparameters"]["img_size"]
learning_rate = config["hyperparameters"]["learning_rate"]
data_dir = config["directory"]["data_dir"]   
result_dir = config["directory"]["result_dir"]   

gpu = config["gpu"]["gpu_index"]

# -------------------------
# データ前処理（データ拡張と正規化）
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# ディレクトリの指定（ImageFolderのディレクトリ構造に準拠）
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

# -------------------------
# デバイス設定（GPUがあれば利用）
device = torch.device(f"cuda:{int(gpu)}" if torch.cuda.is_available() else "cpu")

# -------------------------
# 事前学習済み VGG16 モデルの読み込みとカスタマイズ
model = vgg16(weights=VGG16_Weights.DEFAULT)

# 特徴抽出部分のパラメータは固定
for param in model.features.parameters():
    param.requires_grad = False

# 分類層の最終層を、目的のクラス数に合わせて置き換え
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model = model.to(device)

# -------------------------
# 損失関数、最適化手法、学習率スケジューラの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# -------------------------
# 時間を記録
now = datetime.now()
formatted_time = now.strftime("%Y%m%dT%H%M%S")
print(formatted_time)
result_dir = os.path.join(result_dir, formatted_time)

# -------------------------
# 訓練と検証のループ
best_acc = 0.0
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    print('-' * 10)
    
    # 訓練フェーズと検証フェーズのループ
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        running_corrects = 0
        
        # tqdm を利用した進捗表示
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
        
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # ログの記録
        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.item())
        else:
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc.item())
            # 精度が改善した場合、モデルのチェックポイント（状態辞書）を保存
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                os.makedirs(f"{result_dir}", exist_ok=True)
                torch.save(model.state_dict(), f'{result_dir}/best_model.pth')
                print("Best model saved!")
    
    scheduler.step()
    print()

# -------------------------
# 学習曲線のプロット
plt.figure(figsize=(10, 4))

# 損失 (Loss) のプロット
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 精度 (Accuracy) のプロット
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accs, label='Train Acc')
plt.plot(range(1, epochs+1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 画像をファイルとして保存（PNG形式）
plt.savefig(f'{result_dir}/loss_accuracy_plot.png', dpi=300, bbox_inches='tight')

# 画面に表示
plt.show()

# -------------------------
# ※ 必要に応じて、学習終了後に最終モデルの状態辞書を保存する場合は以下のように記述
# torch.save(model.state_dict(), f'{result_dir}/final_model.pth')
