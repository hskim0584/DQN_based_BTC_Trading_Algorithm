""" State 형식 :
time,   open,   high,   low,    close,  Volume, Volume MA,  RSI,    RSI-based MA,   Upper Bollinger Band,   Lower Bollinger Band
0,      1,      2,      3,      4,      5,      6,          7,      8,              9,                      10,
사용할 것 -> 1, 2, 3, 4, 5, 6, 7, 8
"""
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import numpy as np



class Env:
    def __init__(self, Train=True, maxtime=200, wallet=1000, Unit_size=0.2):
        self.Train = Train
        self.max_episode = 2000
        self.episode = 0

        self.max_time = maxtime
        self.time = 0

        self.done = False
        # 데이터가 들어가 있는 경로로 설정하기
        self.history = pd.read_csv(
            filepath_or_buffer="./BINANCE.csv", encoding="ms932",
            sep=",")
        self.scaler = StandardScaler()
        self.feature = ['open', 'high', 'low', 'close', 'Volume', 'Volume MA', 'RSI', 'RSI-based MA']
        self.scaler.fit(self.history[self.feature])
        self.history_df = self.scaler.transform(self.history[self.feature][:2793])
        print(np.shape(self.history_df))
        self.history_length = 50
        self.min = 931
        self.day = 30
        # 수정

        self.start_num = 0

        # state 정의에 필요한 것들.
        self.state = [[], [], [], [], [], [], [], [], [], [], []]
        self.state2 = [[], [], [], [], [], [], [], [], [], [], []]

        self.Leverage = 0
        # -25, -10, 5, 0 ,5, 10, 25

        self.Possession = 0
        # Possession : 구매 한 비트코인의 개수 ( 음수 가능 ) [단위 : BTC]

        self.Unit_price = 0
        # Unit_price : 구매 한 비트코인의 평단 가격 [단위 : USDT]

        self.Wallet = wallet
        # 현재 지갑에 가진 총 가격 [단위 : USDT]
        # 가용 자산 (현금)

        self.Unit_size = Unit_size
        # 한 번 구매 또는 판매의 단위 [단위 : BTC]

        self.Possession_count = 0

        # 즉, 현재 포지션에 가지고 있는 총 가격은 : Posession * Unit_price,
        # 지갑에 가지고 있는 총 가격은 : Wallet
        # 나의 전체 재산 : Possession * Unit_price + Wallet

    def reset(self):
        if self.Train:
            """ 처음 시작하거나, episode 가 끝이 났을 때,
                    1. state 를 초기화해주고,
                    2. time 또한 초기화해주고,( 0 으로 )
                    3. episode += 1 해준다.
                    출력 : state ( 초기화 된 )"""

            self.Leverage = 0
            # -25, -10, 5, 0 ,5, 10, 25

            self.Possession = 0
            # Possession : 구매 한 비트코인의 개수 ( 음수 가능 ) [단위 : BTC]

            self.Unit_price = 0
            # Unit_price : 구매 한 비트코인의 평단 가격 [단위 : USDT]

            self.Wallet = 1000
            # 현재 지갑에 가진 총 가격 [단위 : USDT]
            # 가용 자산 (현금)

            self.state = [[], [], [], [], [], [], [], [], [], [], []]

            # 1.
            day = random.randrange(1, 3) - 1
            self.start_num = day * self.min + random.randrange(1, self.min - 250)
            # randrange 추후에 csv파일 확인하고 변경
            for i in range(self.history_length):
                self.state[0].append(self.history['open'][self.start_num + i])
                self.state[1].append(self.history['high'][self.start_num + i])
                self.state[2].append(self.history['low'][self.start_num + i])
                self.state[3].append(self.history['close'][self.start_num + i])
                self.state[4].append(self.history['Volume'][self.start_num + i])
                self.state[5].append(self.history['Volume MA'][self.start_num + i])
                self.state[6].append(self.history['RSI'][self.start_num + i])
                self.state[7].append(self.history['RSI-based MA'][self.start_num + i])
                self.state[8].append(self.Wallet)
                self.state[9].append(self.Unit_price)
                self.state[10].append(self.Leverage)
            for i in range(self.history_length):
                self.state2[0].append(self.history_df[self.start_num + i][0])
                self.state2[1].append(self.history_df[self.start_num + i][1])
                self.state2[2].append(self.history_df[self.start_num + i][2])
                self.state2[3].append(self.history_df[self.start_num + i][3])
                self.state2[4].append(self.history_df[self.start_num + i][4])
                self.state2[5].append(self.history_df[self.start_num + i][5])
                self.state2[6].append(self.history_df[self.start_num + i][6])
                self.state2[7].append(self.history_df[self.start_num + i][7])
                self.state2[8].append(self.Wallet)
                self.state2[9].append(self.Unit_price)
                self.state2[10].append(self.Leverage)
            # print(np.shape(self.state2))

            # 2.
            self.time = 0

            # 3.
            self.episode += 1
            self.done = False

            return self.state, self.state2
        else:
            """ 처음 시작하거나, episode 가 끝이 났을 때,
                    1. state 를 초기화해주고,
                    2. time 또한 초기화해주고,( 0 으로 )
                    3. episode += 1 해준다.
                    출력 : state ( 초기화 된 )"""

            self.Leverage = 0
            # -25, -10, 5, 0 ,5, 10, 25

            self.Possession = 0
            # Possession : 구매 한 비트코인의 개수 ( 음수 가능 ) [단위 : BTC]

            self.Unit_price = 0
            # Unit_price : 구매 한 비트코인의 평단 가격 [단위 : USDT]

            self.Wallet = 1000
            # 현재 지갑에 가진 총 가격 [단위 : USDT]
            # 가용 자산 (현금)

            self.state = [[], [], [], [], [], [], [], [], [], [], []]

            # 1.
            self.start_num = 0
            # randrange 추후에 csv파일 확인하고 변경
            for i in range(self.history_length):
                self.state[0].append(self.history['open'][self.start_num + i])
                self.state[1].append(self.history['high'][self.start_num + i])
                self.state[2].append(self.history['low'][self.start_num + i])
                self.state[3].append(self.history['close'][self.start_num + i])
                self.state[4].append(self.history['Volume'][self.start_num + i])
                self.state[5].append(self.history['Volume MA'][self.start_num + i])
                self.state[6].append(self.history['RSI'][self.start_num + i])
                self.state[7].append(self.history['RSI-based MA'][self.start_num + i])
                self.state[8].append(self.Wallet)
                self.state[9].append(self.Unit_price)
                self.state[10].append(self.Leverage)
            for i in range(self.history_length):
                self.state2[0].append(self.history_df[self.start_num + i][0])
                self.state2[1].append(self.history_df[self.start_num + i][1])
                self.state2[2].append(self.history_df[self.start_num + i][2])
                self.state2[3].append(self.history_df[self.start_num + i][3])
                self.state2[4].append(self.history_df[self.start_num + i][4])
                self.state2[5].append(self.history_df[self.start_num + i][5])
                self.state2[6].append(self.history_df[self.start_num + i][6])
                self.state2[7].append(self.history_df[self.start_num + i][7])
                self.state2[8].append(self.Wallet)
                self.state2[9].append(self.Unit_price)
                self.state2[10].append(self.Leverage)
            # 2.
            self.time = 0

            # 3.
            self.episode += 1
            self.done = False

            return self.state, self.state2

    def step(self, action):
        # 입력 변수로 action 을 받음
        # 이 직전의 시장의 가격과 정보로 action 을 내렸고, (구매 또는 판매 완료)
        # 다음 state 는 결국 현재 진행 중인 state. 이게 완료 되면  다음 상태의 가격과 정보는
        # 이미 알고 있다.
        """
        1. 현재 state 에서 action 으로 변화되는 (next_state)
        2. 그 step 에서 받게 되는 (reward)
        3. 다음 (time) += 1
        4. episode 의 끝남 여부 boolean (done)
        출력 : next_state, reward, done, time """

        # print("\n\n================== step ==================")
        reward = 0

        # 1. 2.
        self.state[0].pop(0)
        self.state[1].pop(0)
        self.state[2].pop(0)
        self.state[3].pop(0)
        self.state[4].pop(0)
        self.state[5].pop(0)
        self.state[6].pop(0)
        self.state[7].pop(0)
        self.state[8].pop(0)
        self.state[9].pop(0)
        self.state[10].pop(0)
        self.state[0].append(self.history['open'][self.start_num + 50 + self.time])
        self.state[1].append(self.history['high'][self.start_num + 50 + self.time])
        self.state[2].append(self.history['low'][self.start_num + 50 + self.time])
        self.state[3].append(self.history['close'][self.start_num + 50 + self.time])
        self.state[4].append(self.history['Volume'][self.start_num + 50 + self.time])
        self.state[5].append(self.history['Volume MA'][self.start_num + 50 + self.time])
        self.state[6].append(self.history['RSI'][self.start_num + 50 + self.time])
        self.state[7].append(self.history['RSI-based MA'][self.start_num + 50 + self.time])

        self.state2[0].pop(0)
        self.state2[1].pop(0)
        self.state2[2].pop(0)
        self.state2[3].pop(0)
        self.state2[4].pop(0)
        self.state2[5].pop(0)
        self.state2[6].pop(0)
        self.state2[7].pop(0)
        self.state2[0].append(self.history_df[self.start_num + 50 + self.time][0])
        self.state2[1].append(self.history_df[self.start_num + 50 + self.time][1])
        self.state2[2].append(self.history_df[self.start_num + 50 + self.time][2])
        self.state2[3].append(self.history_df[self.start_num + 50 + self.time][3])
        self.state2[4].append(self.history_df[self.start_num + 50 + self.time][4])
        self.state2[5].append(self.history_df[self.start_num + 50 + self.time][5])
        self.state2[6].append(self.history_df[self.start_num + 50 + self.time][6])
        self.state2[7].append(self.history_df[self.start_num + 50 + self.time][7])
        #state 8, 9, 10 추가

        # 이번 state : state[x][98]
        # 그 상태에서 action 선택
        # 그 이후에 나올 state : state[x][99]

        """
        action = 0 : stay
        action = 1 : Long x5
        action = 2 : Long x10
        action = 3 : Long x25
        action = 4 : Short x5
        action = 5 : Short x10
        action = 6 : Short x25
        action = 7 : Sell All
        action = 8 : Buy more
        action = 9 : Sell once
        """
        pre_Wallet = self.Wallet + self.Unit_price



        # action = 0 : stay
        if action == 0:
            self.state[8].append(self.Wallet)
            if(self.state[3][48] <= self.state[3][49]):
                self.Unit_price = self.Unit_price * (1 + (self.state[3][49]/self.state[3][48] - 1)*self.Leverage)
            else:
                self.Unit_price = self.Unit_price * (1 - (1 - self.state[3][49] / self.state[3][48]) * self.Leverage)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # action = 1 : Long x5
        if action == 1:
            self.Leverage = 5
            if self.Wallet >= 250:
                self.Unit_price = 250
                self.Wallet -= 250
            elif self.Wallet < 250 :
                self.Unit_price = self.Wallet
                self.Wallet = 0

            if (self.state[3][48] <= self.state[3][49]):
                self.Unit_price = self.Unit_price * (1 + (self.state[3][49] / self.state[3][48] - 1) * self.Leverage)
            else:
                self.Unit_price = self.Unit_price * (1 - (1 - self.state[3][49] / self.state[3][48]) * self.Leverage)
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # action = 2 : Long x10
        if action == 2:
            self.Leverage = 10
            if self.Wallet >= 250:
                self.Unit_price = 250
                self.Wallet -= 250
            elif self.Wallet < 250:
                self.Unit_price = self.Wallet
                self.Wallet = 0
            if (self.state[3][48] <= self.state[3][49]):
                self.Unit_price = self.Unit_price * (1 + (self.state[3][49] / self.state[3][48] - 1) * self.Leverage)
            else:
                self.Unit_price = self.Unit_price * (1 - (1 - self.state[3][49] / self.state[3][48]) * self.Leverage)
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet


        # action = 3 : Long x25
        if action == 3:
            self.Leverage = 25
            if self.Wallet >= 250:
                self.Unit_price = 250
                self.Wallet -= 250
            elif self.Wallet < 250:
                self.Unit_price = self.Wallet
                self.Wallet = 0
            if (self.state[3][48] <= self.state[3][49]):
                self.Unit_price = self.Unit_price * (1 + (self.state[3][49] / self.state[3][48] - 1) * self.Leverage)
            else:
                self.Unit_price = self.Unit_price * (1 - (1 - self.state[3][49] / self.state[3][48]) * self.Leverage)
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # action = 4 : Short x5
        if action == 4:
            self.Leverage = -5
            if self.Wallet >= 250:
                self.Unit_price = 250
                self.Wallet -= 250
            elif self.Wallet < 250:
                self.Unit_price = self.Wallet
                self.Wallet = 0
            if (self.state[3][48] <= self.state[3][49]):
                self.Unit_price = self.Unit_price * (1 + (self.state[3][49] / self.state[3][48] - 1) * self.Leverage)
            else:
                self.Unit_price = self.Unit_price * (1 - (1 - self.state[3][49] / self.state[3][48]) * self.Leverage)
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # action = 5 : Short x10
        if action == 5:
            self.Leverage = -10
            if self.Wallet >= 250:
                self.Unit_price = 250
                self.Wallet -= 250
            elif self.Wallet < 250:
                self.Unit_price = self.Wallet
                self.Wallet = 0
            if (self.state[3][48] <= self.state[3][49]):
                self.Unit_price = self.Unit_price * (1 + (self.state[3][49] / self.state[3][48] - 1) * self.Leverage)
            else:
                self.Unit_price = self.Unit_price * (1 - (1 - self.state[3][49] / self.state[3][48]) * self.Leverage)
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # action = 6 : Short x25
        if action == 6:
            self.Leverage = -25
            if self.Wallet >= 250:
                self.Unit_price = 250
                self.Wallet -= 250
            elif self.Wallet < 250:
                self.Unit_price = self.Wallet
                self.Wallet = 0
            if (self.state[3][48] <= self.state[3][49]):
                self.Unit_price = self.Unit_price * (1 + (self.state[3][49] / self.state[3][48] - 1) * self.Leverage)
            else:
                self.Unit_price = self.Unit_price * (1 - (1 - self.state[3][49] / self.state[3][48]) * self.Leverage)
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # action = 7 : Sell All
        if action == 7:
            self.Leverage = 0
            self.Wallet += self.Unit_price
            self.Unit_price = 0
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # action = 8 : Buy more
        if action == 8:
            if self.Wallet <= 250:
                self.Unit_price += self.Wallet
                self.Wallet = 0
            else:
                self.Unit_price += 250
                self.Wallet -=250
            if (self.state[3][48] <= self.state[3][49]):
                self.Unit_price = self.Unit_price * (1 + (self.state[3][49] / self.state[3][48] - 1) * self.Leverage)
            else:
                self.Unit_price = self.Unit_price * (1 - (1 - self.state[3][49] / self.state[3][48]) * self.Leverage)
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # action = 9 : Sell once
        if action == 9:
            if self.Unit_price <= 250:
                self.Wallet += self.Unit_price
                self.Unit_price = 0
                self.Leverage = 0
            else:
                self.Unit_price -=250
                self.Wallet += 250
            self.state[8].append(self.Wallet)
            self.state[9].append(self.Unit_price)
            self.state[10].append(self.Leverage)
            reward = self.Wallet + self.Unit_price - pre_Wallet

        # 3.
        self.time += 1
        # 4.
        if self.time == self.max_time:
            self.done = True
            if self.Train:
                print("Wallet : ", self.Wallet, " Unitprice : ", self.Unit_price)
                print("Sum : ", self.Wallet + self.Unit_price)
        elif pre_Wallet <= 0:
            self.done = True
            if self.Train:
                print("Wallet : 0 \nUnitprice : 0 ㅠㅠㅠ")
            reward = -100
        else:
            self.done = False

        # print("time : ", self.time, "|| done : ", self.done)
        # print(self.episode)
        # print(reward)

        return self.state, self.state2, reward, self.done, self.time