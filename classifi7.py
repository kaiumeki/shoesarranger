# -*- coding: utf-8 -*-
import gpu_config
gpu_config.set_tensorflow([2])

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os


dataset = np.load("dataset6.npz")
x_train, y_train, x_valid, y_valid, x_test, y_test = dataset['X_train'],dataset['y_train'],dataset['X_valid'],dataset['y_valid'],dataset['X_test'],dataset['y_test']

datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,)
# datagen.fit(x_train)
print('train data size:', x_train.shape)
print('valid data size:', x_valid.shape)
print('test data size:', x_test.shape)



# モデルを生成してニューラルネットを構築
model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(100, 100, 3)))
model.add(Activation("relu"))
model.add(Conv2D(64, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(64, 3, 3))
model.add(Activation("relu"))
model.add(Conv2D(128, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(256, 3, 3))
model.add(Activation("relu"))
model.add(Conv2D(512, 3, 3))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())#Flattenだと大きすぎる可能性がある

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(2))
model.add(Activation("softmax"))
model.summary()

# オプティマイザにAdamを使用
opt = Adam(lr=0.0001)
# モデルをコンパイル
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# 学習を実行。10%はテストに使用。
model.fit_generator(datagen.flow(x_train, y_train, batch_size=16), steps_per_epoch=len(x_train)//16, validation_data=[x_valid, y_valid], nb_epoch=100)


print(model.evaluate(x_test, y_test))
print("done")
model.save("new_modeal24.h5")
