# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Convolution2D, Flatten, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os


# 学習用のデータを作る.
image_list = []
label_list = []
kind_list = []

# ./data/train 以下のgood,badディレクトリ以下の画像を読み込む。
for dir in os.listdir("/Users/kai/Desktop/keras2/datafinal"):
    print(dir)
    if dir == ".DS_Store":
        continue
    dir1 = "/Users/kai/Desktop/keras2/datafinal/" + dir#data/train/good or bad
    label = 0 # 一旦，0と設定しておく

    if dir == "goods":    # キレイな靴はラベル0 まだラベルリストに入れていない
        label = 0
    elif dir == "bads":   # 汚い靴はラベル1 まだラベルリストに入れていない
        label = 1

    kind = 0
    for i, file in enumerate(os.listdir(dir1)): #data/train/good or bad/画像ファイル
        if (i-2) % 300 == 0:
            kind += 1
        if file != ".DS_Store":
            # 配列label_listに正解ラベルを追加(キレイな靴:0 汚い靴:1)
            label_list.append(label)
            filepath = dir1 + "/" + file

      #画像パート

            # 画像を100x100pixelに変換し、1要素が[R,G,B]3要素を含む配列の100x100の２次元配列として読み込む。
            # [R,G,B]はそれぞれが0-255の配列。
            image = np.array(Image.open(filepath).resize((100, 100)))
            #print(filepath) #進行状況の表示
            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
#            image = image.transpose(2, 0, 1)
            #print(image.shape) #進行状況の表示
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.) #なんで255で割るのかがわからない
            if i % 50 == 0:
                print(i)
            kind_list.append(kind)

import collections

# kerasに渡すためにnumpy配列に変換。
image_list = np.array(image_list)
print(kind_list)
print(collections.Counter(kind_list))
# ラベルの配列を1と0からなるラベル配列に変更
# 0 -> [1,0], 1 -> [0,1] という感じ。
#Y = to_categorical(label_list)

# np.savez_compressed('finaldataset.npz', data=image_list, label=np.array(label_list))
