'''
http://torch.classcat.com/2018/05/12/pytorch-tutorial-intermediate-char-rnn-classification/
文字レベル(≠単語レベル)RNNによる名前分類
18の言語(日本語含む！)から合計数千個の名字によって学習し、入力した名前がどの言語であるかを予測する

準備：ここからデータをダウンロードする。
https://download.pytorch.org/tutorial/data.zip
'''
from __future__ import unicode_literals
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def findFiles(path): return glob.glob(path)
print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
print(all_letters) #a...zA...z .,;
n_letters = len(all_letters)
print(n_letters) #57文字

# Unicode -> ASCII---------------------------------------------------
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
print('Ślusàrski')
print(unicodeToAscii('Ślusàrski'))

# Make name lists of each languages----------------------------------
category_lines = {} #言語ごとの名前のリストの辞書
all_categories = [] #カテゴリ（=言語）のリスト

def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

print(category_lines['Italian'][3:5]) #['Abate', 'Abategiovanni']
print(category_lines['Japanese'][:1]) #['Abe']

#NameToTensor--------------------------------------------------------
# 文字のインデックスを返す e.g. "a"=0, 'c'=2
def letterToIndex(letter):
    return all_letters.find(letter)

# 文字単位のone-hotベクトル作成
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor
print(letterToTensor('a')) # tensor([[1., 0., ... 0.]])

# 単語（名前）単位のone-hotベクトル作成（文字数×1×語彙数)
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor
print(lineToTensor('Abe'))

#defineRNN-----------------------------------------------------------
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
#generateRNN---------------------------------------------------------
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input = lineToTensor('Abe')
hidden = torch.zeros(1, n_hidden)

output, next_hidden = rnn(input[0], hidden) #forwardが呼び出される
print(output) #1×言語数のTensor;言語の尤もらしさを意味

#prepare HelperFunctions---------------------------------------------
# 最も尤もらしい言語とそのインデックスを返す
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output)) #('Arabic', 2)

# データをランダムに抽出する
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]
def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)

#defineTrain---------------------------------------------------------
criterion = nn.NLLLoss() #損失関数;クロスエントロピー誤差

learning_rate = 0.005 #学習率

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]): #1文字ずつRNNに入力
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor) #損失計算
    loss.backward() #勾配計算

    # 重みの更新
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

#executeTrain--------------------------------------------------------
# オンライン学習(1データごとに更新)
current_loss = 0
all_losses = []

n_iters = 100000 #エポック数
print_every = 5000
plot_every = 1000

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # エポック数 進行率% (学習時間) 損失 入力 / 出力 ラベル(正しければ'✓')
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

#plotLosses----------------------------------------------------------
plt.figure()
plt.plot(all_losses)
plt.show()
