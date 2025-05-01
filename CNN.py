import random

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CNN:
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None):
        np.random.seed(7)
        self.x_test = x_test
        self.x_train = x_train
        self.y_test = y_test
        self.y_train = y_train
        self.input_shape = (8, 50, 1)
        self.model = None

    def train(self):
        EPOCH = 100
        BATCH_SIZE = 20

        # print('x_train shape:', self.x_train.shape)
        # print(self.x_train.shape[0], 'train samples')
        # print(self.x_test.shape[0], 'test samples')

        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(4, 4), activation='relu', input_shape=self.input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dense(500, activation='relu'))
        self.model.add(Dense(1, activation='relu'))
        print(self.model.summary())

        self.model.compile(loss='mse', optimizer='adam', metrics=['mse'])

        history = self.model.fit(self.x_train, self.y_train, batch_size=BATCH_SIZE, epochs=EPOCH, verbose=1, shuffle=True)
        # score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])

        self.model.save('CNNmodel.h5')

        # y_vloss = history.history['val_loss']
        y_loss = history.history['loss']

        # 그래프
        x_len = np.arange(len(y_loss))
        # plt.plot(x_len, y_vloss, marker='.', c="red", label='Testset_loss')
        plt.plot(x_len, y_loss, marker='.', c="blue", label='Trainset_loss')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

        return self.model

    def load_CNN(self, h5):
        # 학습 용도가 아닌, 완성된 모델을 불러 객체를 생성하기 위한 부분
        self.model = tf.keras.models.load_model(h5)


def main():
    train = True
    # train = False
    if train:
        history = pd.read_csv(
            filepath_or_buffer="./BINANCE.csv", encoding="ms932",
            sep=",")
        data = []
        for q in range(1):
            for i in range(200):
                state = [[], [], [], [], [], [], [], []]
                for j in range(50):
                    state[0].append(history['open'][931 * q + j + i + 1])
                    state[1].append(history['high'][931 * q + j + i + 1])
                    state[2].append(history['low'][931 * q + j + i + 1])
                    state[3].append(history['close'][931 * q + j + i + 1])
                    state[4].append(history['Volume'][931 * q + j + i + 1])
                    state[5].append(history['Volume MA'][931 * q + j + i + 1])
                    state[6].append(history['RSI'][931 * q + j + i + 1])
                    state[7].append(history['RSI-based MA'][931 * q + j + i + 1])
                state.append(history['close'][931 * q + i + 1 + 50])
                data.append(state)
        print(data[0][0], ' ', data[0][8])
        # print(np.shape(data))
        y_train = []
        for i in range(200):
            y_train.append(data[i][8])
            data[i].pop(8)
        data = np.reshape(data, (200, 8, 50, 1))
        y_train = np.reshape(y_train, (200, 1))
        print(np.shape(data))
        print(np.shape(y_train))
        A = CNN(data, None, y_train, None)
        A.train()
    else:
        history = pd.read_csv(
            filepath_or_buffer="./BINANCE.csv", encoding="ms932",
            sep=",")
        real = []
        pred = []
        path = "./CNNmodel.h5"
        A = CNN()
        A.load_CNN(path)
        B = 0
        for j in range(100):
            state = [[], [], [], [], [], [], [], []]
            for i in range(50):
                state[0].append(history['open'][i + j])
                state[1].append(history['high'][i + j])
                state[2].append(history['low'][i + j])
                state[3].append(history['close'][i + j])
                state[4].append(history['Volume'][i + j])
                state[5].append(history['Volume MA'][i + j])
                state[6].append(history['RSI'][i + j])
                state[7].append(history['RSI-based MA'][i + j])
            state = np.reshape(state, (1, 8, 50, 1))
            pred.append(A.model.predict(state, verbose=0)[0][0] - B )
            real.append(history['close'][j + 50])
            print("predict : ", A.model.predict(state, verbose=0)[0][0], ",     real : ", history['close'][j + 50])
            # B = A.model.predict(state, verbose=0)[0][0] - history['close'][j + 50]
            if j % 5 == 0:
                B = A.model.predict(state, verbose=0)[0][0] - history['close'][j + 50]
        # plt.plot(A, 'ro')
        # print(count)
        plt.figure()
        plt.plot(pred, 'r', linestyle='--', label='Predicted Close')
        plt.plot(real, 'b', label='Actual Close')
        plt.legend()
        plt.show()

        count = 0
        A = []
        for i in range(99):
            if (real[i + 1] > real[i]) & (pred[i+1] > real[i]):
                count += 1
                A.append(real[i])
            elif (real[i+1] < real[i]) & (pred[i+1] < real[i]):
                count += 1
                A.append(real[i])
            else :
                A.append(None)
        plt.plot(A, 'ro')
        print(count)
        plt.plot(pred, 'r', linestyle='--', label='Predicted Close')
        plt.plot(real, 'b', label='Actual Close')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
