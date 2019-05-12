# About
[Pytorch公式チュートリアル](https://pytorch.org/tutorials/)への取り組みをまとめる. 基本的にコードはそのまま, コメント多めです.


### 1. [NEURAL NETWORKS](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)
- CNNの原形であるLeNetの実装.\\
  ![lenet](https://github.com/fury00812/PytorchTutorials/tree/master/images/lenet.png)

  上記画像のように, 32\*32の画像を受け取り, 28\*28(畳み込み)→14\*14(プーリング)→10\*10(畳み込み)→5\*5(プーリング)→1\*120(全結合)→1\*84(全結合)→1\*10(全結合)と最終的に10次元に圧縮.
  
- モデルの構築から出力・損失計算,パラメータ更新と一連の流れが理解できる. ただ学習は1度切りで繰り返し処理は無い．

- [こちらのサイト](https://qiita.com/mckeeeen/items/e255b4ac1efba88d0ca1)が日本語で丁寧に解説してくださってます.

