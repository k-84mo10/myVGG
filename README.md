# myVGG
このリポジトリはVGGを用いて画像分類のタスクをするためのリポジトリです。

## Install
まず、仮想環境を作成することをオススメします。
```sh
python3 -m venv myenv
source myenv/bin/activate
```
次に、必要なライブラリをインストールしていきます。
```sh
pip install -m requirements.txt
```

## How to Use
このリポジトリには「モデルの学習」と「モデルの可視化」の二つの役割があります。
### モデルの学習
まず、シンボリックリンクなどを用いて、datasetのpathをこのディレクトリに置いてください。
```sh
ln -s <元のpath> <このディレクトリのpath>
```
次に、config.tomlにハイパーパラメータやデータセットのpathを記入してください。  
以下は例です。
```toml
[hyperparameters]
num_classes = 4        # クラス数（例: 4クラス分類）
batch_size = 32
epochs = 10
img_size = 224
learning_rate = 0.001

[directory]
data_dir = 'thai/data/mydesk/mydesk_dataset'    
result_dir = 'thai/result'

[gpu]
gpu_index = 1
```
その後、以下のコードで学習が始まります。
```sh
python3 vgg.py
```
学習したモデル・各種ログなどは`result_dir`内に生成されます。
### モデルの可視化
作成したモデルの可視化は`grad_cam_vgg.py`で行います。  
まず、`grad_cam_vgg.py`のsettingを入力してください。
```py
#--------------------------------------------------------------------------------------#
# setting
num_classes = 4
result_dir = "thai/result/20250301T141823"
model_name = "best_model.pth"
img_path = "thai/data/mydesk/mydesk_raw/run3/20250228T165549_0011.jpg"
#--------------------------------------------------------------------------------------#
```
`result_dir`はモデルが入っているフォルダの名前、`model_name`はモデルの名前を書いてください。  
`img_path`は可視化したい画像のpathです。  
次に、以下のコードを動かすと可視化できます。
```sh
python3 grad_cam_vgg.py
```
`input.jpg`・`heatmp.jpg`・`overlay.jpg`の3枚が`result_dir`内に生成されます。