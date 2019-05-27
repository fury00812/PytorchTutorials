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

def findFiles(path): return glob.glob(path)
print(findFiles('data/names/*.txt'))

all_letters = string.ascii_letters + " .,;'"
print(all_letters) #a...zA...z .,;
n_letters = len(all_letters)
print(n_letters)

# Unicode -> ASCII----------------------------------
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
print('Ślusàrski')
print(unicodeToAscii('Ślusàrski'))

# Make name lists of each languages-----------------
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
#--------------------------------------------------
