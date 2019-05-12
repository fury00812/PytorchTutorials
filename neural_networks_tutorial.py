'''
https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
CNNの原形LeNetの実装
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    '''
    畳み込み(conv1)→プーリング(max_pool2d)→畳み込み(conv2)
        →プーリング(max_pool2d)→全結合(fc1)→全結合(fc2)→全結合(fc3)
    *畳み込みとプーリングの間には活性化関数ReLUを挟む
    '''
    def __init__(self):
        super(Net, self).__init__()

        #畳み込み層の定義
        self.conv1 = nn.Conv2d(1, 6, 5) #入力レイヤー1, 出力レイヤー6, カーネル5*5
        self.conv2 = nn.Conv2d(6, 16, 5) #入力レイヤー6, 出力レイヤー16, カーネル5*5

        #全結合層の定義
        self.fc1 = nn.Linear(16 * 5 * 5, 120) #(16*5*5)行120列の重み
        self.fc2 = nn.Linear(120, 84)  #120行84列の重み
        self.fc3 = nn.Linear(84, 10) #84行10列の重み

    def forward(self, x):
        #入力→conv1→ReLU→プーリング層1(2*2)→出力
        x = F.max_pool2d(F.relu(self.conv1(x)), 2) #次元32→28→14
        #入力→conv2→ReLU→プーリング層2(2*2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) #次元14→10→5

        #全結合層へ繋げるために1次元ベクトル化
        x = x.view(-1, self.num_flat_features(x)) #サイズ[1, 16*5*5]
        x = F.relu(self.fc1(x)) #[1, 120]
        x = F.relu(self.fc2(x)) #[1, 84]
        x = self.fc3(x) #[1,10]
        return x

    def num_flat_features(self, x):
        '''
        特徴量の数を取得
        '''
        #x.size() = torch.Size([サンプル数,レイヤー数16,縦5,横5])
        #x.size()[1:] = torch.Size([16,5,5])
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s #1*16*5*5
        return num_features

if __name__ == '__main__':

    '''
    step1 : モデルのインスタンス生成・確認
    '''
    net = Net()
    print(net)
    # Net(
    #  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
    #   (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
    #   (fc1): Linear(in_features=400, out_features=120, bias=True)
    #   (fc2): Linear(in_features=120, out_features=84, bias=True)
    #   (fc3): Linear(in_features=84, out_features=10, bias=True)
    # )

    #params=[conv1's weight, conv1's bias, conv2's weight, conv2's bias,
    #    fc1's weight, fc1's bias, fc2's weight, fc2's bias, fc3's weight, fc3's bias]
    params=list(net.parameters())
    print(len(params)) #10

    #conv1の重みのサイズ
    print(params[0].size()) #torch.Size([6, 1, 5, 5])
    #conv2の重みのサイズ
    print(params[1].size()) #torch.Size([6])

    '''
    step2 : ランダムな32*32画像で計算してみる
    '''
    input = torch.randn(1, 1, 32, 32) #サンプル数*チャネル数*縦の長さ*横の長さ
    #nn.Moduleクラスのインスタンスに引数が渡されるどforwardを実行して結果を返す
    output = net(input)
    print(output)
    # tensor([[ 0.1385, -0.0148,  0.0199, -0.0067,  0.0235, -0.0743,
    #     -0.0432,  0.0986,　-0.0185,  0.0626]], grad_fn=<AddmmBackward>)

    '''
    step3 : パラメータの勾配を初期化し,出力に対する各パラメータの勾配を再計算
    '''
    net.zero_grad()
    output.backward(torch.randn(1, 10)) #パラメータが複数個の場合,その数に応じて適当な引数を与える必要がある
    print(params[0].grad.data) #6チャネルの各カーネル(5*5)が表示される

    '''
    step4 : ランダムな教師データを作成し, 出力との損失を計算
    '''
    output = net(input) #出力
    target = torch.randn(10)  #適当な教師データ
    target = target.view(1, -1)  # 出力と形を揃える
    criterion = nn.MSELoss() #損失関数(平均二乗誤差)のインスタンスを生成
    loss = criterion(output, target) #損失を計算
    print(loss) #tensor(0.9425, grad_fn=<MseLossBackward>)

    '''
    step5 : 損失に対する各パラメータの勾配を計算
    '''
    net.zero_grad() #勾配の初期化
    #conv1のバイアスが初期化されていることを確認
    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward() #勾配の計算
    #conv1が計算されていることを確認
    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

    '''
    step6 : 勾配降下法で重みを更新
    '''
    learning_rate=0.01 #学習率
    for f in net.parameters():
        #f.data -= f.data - learning_rate*f.grad.data
        f.data.sub_(learning_rate*f.grad.data)
    print(params[0].grad.data)

    '''
    step7 : 最適化モジュールによる実装（step4~6）
    '''
    #SGDのインスタンスを生成
    optimizer=optim.SGD(net.parameters(),lr=0.01)

    #イテレーション毎に以下を実行する
    optimizer.zero_grad() #勾配の初期化
    output=net(input) #出力の計算
    loss=criterion(output, target) #lossの計算
    loss.backward() #勾配の計算(誤差逆伝播)
    optimizer.step() #重みの更新
    print(params[0].grad.data)
