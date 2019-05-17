'''
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
PyTorchの売りであるautograd（自動微分）についてその使い方を紹介
・print("----")で区切ってjupyterやcolaboratoryとかで動かすと分かりやすいかもしれません
・一部プログラムを改変しています
'''
import torch

print("----------")
#requires_grad=True : 計算グラフを保持 i.e.そのVariableの全ての演算を追跡する
#これにより, backward()を使って全ての勾配を自動的に計算できる
x = torch.ones(2,2,requires_grad=True)
print(x)
# tensor([[1., 1.],
#         [1., 1.]], requires_grad=True)

#--------------------------------------------------
#演算によって生成されたvariableには属性grad_fnが付与される
#※独立変数xのrequires_grad=Trueの時のみ
y = x + 2
print(y)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)

#--------------------------------------------------
#属性grad_fn : どの演算(function)によってvariableが生成されたか？
print(y.grad_fn)
# <AddBackward0 object at 0x101b40e48>

#--------------------------------------------------
#演算によって生成されたvariableには属性grad_fnが付与される
z = y * y * 3   #"Mul"backward
out = z.mean()  #"Mean"backward
print(z)
print(out)
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward0>)
# tensor(27., grad_fn=<MeanBackward0>)

#--------------------------------------------------
#.requires_grad_(...)を使用すると既存のvariableのrequires_gradフラグ（前述）を変更できる.
#このフラグはデフォルトではFalseである.
a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad) #False
a.requires_grad_(True)
print(a.requires_grad) #True

#いよいよ勾配計算------------------------------------------
out.backward() #out.backward(torch.tensor(1.))と等価
print(x.grad) #outのxに関する勾配
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])

#--------------------------------------------------
#no_grad()ブロック内に記述するとautogradに追跡されない
print(x.requires_grad) #True
print((x ** 2).requires_grad) #True
with torch.no_grad():
    print((x ** 2).requires_grad) #True
print((x ** 2).requires_grad) #True
