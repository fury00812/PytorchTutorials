##ネットワーク関連
###torch.nn.Conv2d
畳み込み層の定義
```python
torch.nn.Conv2d(入力レイヤー, 出力レイヤー, カーネルサイズ)
```

###torch.nn.Linear
線形層($$y=Wx+b$$)の定義
```
torch.nn.Linear(in_features, out_features, bias=True)
```
例えば, 次元数3のデータを2つ用意してこれを5次元に変換する場合
```python
x = torch.randn(2, 3)
W = torch.nn.Linear(3, 5, False)
y = m(x)
print(y.size())
#torch.Size([2, 5])

```
$$
x=
\begin{bmatrix}
x_{11} & x_{12} & x_{13} \\
x_{21} & x_{22} & x_{23}
\end{bmatrix},\\
W=
\begin{bmatrix}
w_{11} & w_{12} & w_{13} & w_{14} & w_{15}\\
w_{21} & w_{22} & w_{23} & w_{24} & w_{25}\\
w_{31} & w_{32} & w_{33} & w_{34} & w_{35}
\end{bmatrix},\\
y=xW = 
\begin{bmatrix}
y_{11} & y_{12} & y_{13} & y_{14} & y_{15}\\
y_{21} & y_{22} & y_{23} & y_{24} & y_{25}
\end{bmatrix}\\
$$

###torch.nn.functional.max_pool2d
プーリング処理を行う**関数**[^1]
```python
torch.nn.functional.max_pool2d(入力, カーネルサイズ)
```



##勾配

###net.zero_grad()
netの各パラメータの勾配を0で初期化（初期化するのは, 勾配がイテレーション毎に加算される仕様であるため）

###backward()
出力や損失に対する各パラメータの勾配を計算する
```python
y.backward()
```
- ```backward()```は```backward(torch.Tensor([1]))```の省略形である. 
そのため, 例えば出力が[1,10]次元のベクトルである場合
```python
y.backward(torch.randn(1, 10))
```
のようにする.



##損失
###nn.MSELoss()
平均二乗誤差関数
```python
criterion=nn.MSELoss()
loss=criterion(output,target)
```



##Tensor関連

###transpose
転置
```python
x = torch.randn(4,3) #4行3列
torch.transpose(x)#3行4列
```
###view
サイズ変更
```python
x = torch.randn(4,3) #4行3列
x.view(12,1) #12行1列
x.view(-1,2) #n行2列（この場合n=6）
```

###size
```python
x = torch.randn(4,3)
x.size() #torch.Size([4,3])
```
#その他用語
- Affine(アフィン) : 全結合層, 線型層

[^1]:プーリング層を定義するクラス(torch.nn.MaxPool2d)もある