'''
https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
テンソルの扱い（初期化,演算,ndarray相互変換など）を紹介
'''
import torch
import numpy as np

#tips1: Tensorを作成する. 作成したTensorは初期化されておらず何らかの値が入っている.
x = torch.empty(5, 3)
print(x)
#tensor([[ 0.0000e+00,  1.5846e+29, -2.6611e-27],
#        [-3.6902e+19, -2.6884e-27,  1.5849e+29],
#        [ 0.0000e+00,  1.5846e+29,  5.6052e-45],
#        [ 0.0000e+00,  0.0000e+00,  1.5846e+29],
#        [ 0.0000e+00,  1.5846e+29, -2.6615e-27]])

#tips2:ランダムなTensorを作成する.
x = torch.rand(5, 3)
print(x)
#tensor([[0.5934, 0.8821, 0.2641],
#        [0.3726, 0.3678, 0.2573],
#        [0.2761, 0.6679, 0.7228],
#        [0.9908, 0.5877, 0.5313],
#        [0.4664, 0.1699, 0.0252]])

#tips3:要素が0で初期化されたTensorを作成する.
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
#tensor([[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]])

#tips4:値を直接指定してTensorを作成する.
x = torch.tensor([5.5, 3])
print(x)
#tensor([5.5000, 3.0000])

#tips5:既存のTensorを上書きする. ちなみに元のshapeを保つ必要は無い.
x = torch.tensor([5.5, 3])
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
#tensor([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.]], dtype=torch.float64)

#tips6: Tensorの値をランダムに書き換え, 型を変換する
# 下の例では5*3のゼロ行列を, 5*3の値がランダムなフロート型に変換
x = x.new_zeros(5, 3, dtype=torch.double)
print(x)
#tensor([[0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0],
#        [0, 0, 0]])
x = torch.randn_like(x, dtype=torch.float)
print(x)
#tensor([[ 0.4074, -0.4801, -1.7652],
#        [ 0.9456, -0.8796, -0.5751],
#        [-1.5125, -2.3049,  0.7683],
#        [ 0.8419, -0.6031, -0.2340],
#        [ 0.7071,  0.8503,  0.5636]])

#tips7: Tensor同士の和
x = torch.ones(5, 3)
y = torch.rand(5, 3)
print(y)
print(x + y)
# tensor([[0.7386, 0.6484, 0.6591],
#         [0.4434, 0.3713, 0.1573],
#         [0.9194, 0.7912, 0.9306],
#         [0.2761, 0.4009, 0.6479],
#         [0.9895, 0.4350, 0.1566]])
# tensor([[1.7386, 1.6484, 1.6591],
#         [1.4434, 1.3713, 1.1573],
#         [1.9194, 1.7912, 1.9306],
#         [1.2761, 1.4009, 1.6479],
#         [1.9895, 1.4350, 1.1566]])

#tips7.1: add関数を使う方法
print(torch.add(x, y))

#tips7.2: 別のTensorに代入する方法
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

#tips7.3: インクリメント的な書き方
y.add_(x)
print(y)

#tips8: numpy的なスライシングもできる
print(y[:, 1])
#tensor([0.6484, 0.3713, 0.7912, 0.4009, 0.4350])

#tips9: リサイズ
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)
print(x.size(), y.size(), z.size())
#torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])

#tips10: 要素数1のTensorの値を得る
x = torch.randn(1)
print(x)
print(x.item())
# tensor([-0.6118])
# -0.6117833256721497

#tips11: Tensorをndarrayに変換
a = torch.ones(5)
print(a)　#tensor([1., 1., 1., 1., 1.])
b = a.numpy()
print(b) #[1. 1. 1. 1. 1.]

#tips11.1: aの値の変更がbにも反映される
a.add_(1)
print(a)
print(b)
# tensor([2., 2., 2., 2., 2.])
# [2. 2. 2. 2. 2.]

#tips12: ndarrayをTensorに変換
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
# [2. 2. 2. 2. 2.]
# tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
