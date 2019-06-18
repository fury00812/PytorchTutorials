# About
[Pytorch公式チュートリアル](https://pytorch.org/tutorials/)への取り組みをまとめる. 基本的にコードはそのまま, コメント多めです.  
以下の紹介は取り組んだ順に追記していきます

### [NEURAL NETWORKS](https://github.com/fury00812/PytorchTutorials/blob/master/neural_networks_tutorial.py)([公式](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html))
- CNNの原形であるLeNetの実装.  
  ![lenet](https://user-images.githubusercontent.com/35480446/57579766-b41d7500-74db-11e9-812c-5883e1a7923f.png)

  上記画像のように, 32\*32の画像を受け取り, 28\*28(畳み込み)→14\*14(プーリング)→10\*10(畳み込み)→5\*5(プーリング)→1\*120(全結合)→1\*84(全結合)→1\*10(全結合)と最終的に10次元に圧縮.
  
- モデルの構築から出力・損失計算,パラメータ更新と一連の流れが理解できる. ただ学習は1度切りで繰り返し処理は無い．

- [こちらのサイト](https://qiita.com/mckeeeen/items/e255b4ac1efba88d0ca1)が日本語で丁寧に解説してくださってます.

### [WHAT IS PYTORCH?](https://github.com/fury00812/PytorchTutorials/blob/master/tensor_tutorial.py)([公式](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html))
- テンソルの扱い方（初期化,演算,ndarray相互変換など）を紹介
- 基本的なTensor作成方法が列挙されている. チュートリアルというかリファレンス的な感じ
- Numpy Arrayとの変換は便利そうだなと思った

### [AUTOGRAD: AUTOMATIC DIFFERENTIATION](https://github.com/fury00812/PytorchTutorials/blob/master/autograd_tutorial.py)([公式](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html))
- PyTorchの売りであるautograd（自動微分）についてその使い方を紹介
- print("----")で区切ってjupyterやcolaboratoryとかで動かすと分かりやすいかもしれません

### [CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://github.com/fury00812/PytorchTutorials/blob/master/pytorch-tutorial-intermediate-char-rnn-classification.py)([公式](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html))
- RNNモジュールを用いた時系列データ処理のチュートリアル. 人名を入力として与え18の言語に分類
- 機械学習の一連の流れが掴める(データの準備, モデルの定義, 学習, 評価)
  ![rnn_loss](https://user-images.githubusercontent.com/35480446/58886668-4867aa00-871f-11e9-9746-0a87f5dff15b.png)
