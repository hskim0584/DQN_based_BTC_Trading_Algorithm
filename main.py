import random
import numpy as np
from collections import deque

import pandas as pd
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from keras.optimizers import Adam
import tensorflow as tf
from env import Env
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ENV_NAME = ''  # Environment name
FRAME_WIDTH = 14  # Resized frame width (chart + wallet)
FRAME_HEIGHT = 50  # Resized frame height
NUM_EPISODES = 10000  # Number of episodes the agent plays
# NUM_EPISODES = 30  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
# 0.99 ~ 0.9
EXPLORATION_STEPS = 500000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 20000 # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 40000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size (11*50*4를 32개 : 묶어서 학습할 때 사용)
TARGET_UPDATE_INTERVAL = 5000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.001  # Learning rate used by RMSProp
# 0.001 ~ 0.01
MOMENTUM = 0.99  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 10000  # The frequency with which the network is saved
NO_OP_STEPS = 10  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
# LOAD_NETWORK = False
LOAD_NETWORK = True
TRAIN = False
# TRAIN = True

NUM_EPISODES_AT_TEST = 10  # Number of episodes the agent plays at test time

# feed_dict에서 255로 나눈 거 다 없앴음.
# 하이퍼파라미터는 그대로 갖고 가기.
# 데이터 종류별로 정규화 하는 법 찾기.

# 딥러닝 모델 제작
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__() # keras.Model 상속
        self.h0 = Conv2D(32, kernel_size=(4, 4), activation='relu')
        self.m0 = MaxPooling2D(pool_size=(2, 2))
        self.b0 = BatchNormalization()

        self.h1 = Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.m1 = MaxPooling2D(pool_size=(2, 2))
        self.b1 = BatchNormalization()

        self.h2 = Conv2D(128, kernel_size=(2, 2), activation='relu')
        self.m2 = MaxPooling2D(pool_size=(2, 2))
        self.b2 = BatchNormalization()

        self.d1 = Dropout(0.2)
        self.F = Flatten()
        self.d2 = Dropout(0.2)

        self.h3 = Dense(1000, activation='relu')
        self.h4 = Dense(500, activation='relu')
        self.out = Dense(action_size, activation='softmax',
                         kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1e-3, maxval=1e-3))

    def call(self, state):
        x = self.h0(state)
        # x = self.m0(x)
        # x = self.b0(x)

        x = self.h1(x)
        # x = self.m1(x)
        # x = self.b1(x)

        x = self.h2(x)
        # x = self.m2(x)
        # x = self.b2(x)

        # x = self.d1(x)
        x = self.F(x)
        # x = self.d2(x)

        x = self.h3(x)
        x = self.h4(x)
        out = self.out(x)

        return out

class Agent():
    def __init__(self, num_actions):
        self.newstate = None
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()
        # Create q network
        self.state_size = (FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH)
        self.behavior_model = DQN(self.num_actions)  # replay 메모리 저장용
        self.target_model = DQN(self.num_actions)  # 정책 학습용

        self.behavior_model.build(input_shape=(None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH))
        self.target_model.build(input_shape=(None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH))

        self.behavior_model.summary()
        self.target_model.summary()

        self.optimizer = Adam(learning_rate=LEARNING_RATE)
        self.newstate_df = None

        # 타깃 모델 초기화
        self.update_target_model()
        self.f = open('reward.txt', 'w', encoding='utf-8', newline='\n')


    def update_target_model(self):
        weights = self.behavior_model.get_weights()
        self.target_model.set_weights(weights)

    # 여기서 state 전처리 하기 (normalization or standardization)
    def get_initial_state(self, observation): # t : 0~3 observation -> initial_state (20*50*4)
        state = [observation for _ in range(STATE_LENGTH)] # 같은 시간대만 4개? ㅇㅇ
        return np.stack(state, axis=0) # (4, 11, 50)

    def get_action(self, state, state2):
        # state2 = [open, high, low, close, volume, volume MA, RSI, RSI MA, wallet, unit_price, leverage]
        # action = [stay, long5, long10, long25, short5, short10, short25, sell all, buy, sell]
        # 250 이하일 땐, 남은 돈 전체를 써서 사게끔 수정될 듯
        possible_action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # wallet == 0 : remove 1~6, 8
        # 지갑 0, unit_price != 0 , leverage != 0
        if state2[8][49] == 0 or state2[9][49] != 0 or state2[10][49] != 0:
            possible_action.remove(1)
            possible_action.remove(2)
            possible_action.remove(3)
            possible_action.remove(4)
            possible_action.remove(5)
            possible_action.remove(6)

        if state2[8][49] == 0 or state2[10][49] == 0:
            possible_action.remove(8)

        # unit_price == 0 : remove 7,9
        if state2[9][49] == 0 or state2[10][49] == 0:
            possible_action.remove(7)
            possible_action.remove(9)


        poss_q_value = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
        for i in possible_action:
            poss_q_value[0][i] = float(self.behavior_model(state)[0][i])
        # print(poss_q_value)
            # 실제 딥러닝 input : (20*50*4)

        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = np.random.choice(possible_action)
        else:
            action = np.nanargmax(poss_q_value[0])
            # print(self.q_values.eval(feed_dict={self.s: [np.float32(state)]})[0])
            # print(poss_q_value)

        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[:, :, :, 1:], observation, axis=3)
        # print("aaaaa",next_state.shape)
        # print("bbbbb", state.shape)
        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -3, 3)

        # Store transition in replay memory (terminal = end 1 episode)
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                weights = self.behavior_model.get_weights()
                self.target_model.set_weights(weights)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                self.behavior_model.save_weights("./DQNModel.h5")
                print('Successfully saved: DQNModel.h5')

        self.total_reward += reward
        self.duration += 1

        if terminal:
            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print(
                'EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_LOSS: {5:.5f} / MODE: {6}'.format(
                    self.episode + 1, self.t, self.duration, self.epsilon,
                    self.total_reward,
                    self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))
            if (self.episode % 5) == 0:
                self.f.write('%f ' %self.total_reward)
                self.f.write('%f \n' %self.total_loss)
                print("\nreward save successfully : reward.txt\n")

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1


        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        model_params = self.behavior_model.trainable_variables
        # print(np.shape(state_batch))
        state_batch = np.reshape(state_batch, (32, 14, 50, 4)) # reshape 이전의 형태는?
        # print(np.shape(state_batch))
        next_state_batch = np.reshape(next_state_batch, (32, 14, 50, 4))
        with tf.GradientTape() as tape:
            # 현재 상태에 대한 모델의 큐함수
            predicts = self.behavior_model(state_batch)
            one_hot_action = tf.one_hot(action_batch, self.num_actions)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)
            target_predicts = self.target_model(next_state_batch)
            target_predicts = tf.stop_gradient(target_predicts) # target net 은 학습을 바로 안 시킴 (나중에 update)

            # 벨만 최적 방정식을 이용한 업데이트 타깃
            terminal_batch = np.array(terminal_batch) + 0
            max_q = np.amax(target_predicts, axis=-1) # target에서 max_q batch 추출
            # 여기가 Greedy policy로 off policy인 부분임
            targets = reward_batch + (1 - terminal_batch) * GAMMA * max_q # episode가 끝나는 부분은?
            loss = tf.reduce_mean(tf.square(targets - predicts))  # loss 함수로 mse 사용 (reduce_mean : 전체 평균)
            self.total_loss = self.total_loss + loss
            # 오류 함수를 줄이는 방향으로 모델 업데이트
            grads = tape.gradient(loss, model_params) # 자동 미분
            self.optimizer.apply_gradients(zip(grads, model_params)) # 파라미터 업데이트

    def load_network(self):
        self.behavior_model.load_weights("./DQNModel.h5")
        self.target_model.load_weights("./DQNModel.h5")
        print('Successfully loaded: DQNModel.h5')

    def get_action_at_test(self, state, state2):
        possible_action = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # wallet <= 250 : remove 1~6
        # wallet <= 0 : remove 1~6 + 8
        if state2[8][49] == 0:
            possible_action.remove(1)
            possible_action.remove(2)
            possible_action.remove(3)
            possible_action.remove(4)
            possible_action.remove(5)
            possible_action.remove(6)
            possible_action.remove(8)

        # unit_price == 0 : remove 7,9
        if state2[9][49] == 0 or state2[10][49] == 0:
            possible_action.remove(7)
            possible_action.remove(8)
            possible_action.remove(9)

        poss_q_value = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
        for i in possible_action:
            poss_q_value[0][i] = float(self.behavior_model(state)[0][i])

        if random.random() <= 0.01:
            action = np.random.choice(possible_action)
        else:
            action = np.nanargmax(poss_q_value[0])
        return action
    def make_newstate(self, pre_state, observation2): # step 마다 사용
        for i in range(14):
            self.newstate[i].pop(0)
        self.newstate[0].append(observation2[0][49])
        self.newstate[1].append(observation2[0][49])
        self.newstate[2].append(observation2[1][49])
        self.newstate[3].append(observation2[1][49])
        self.newstate[4].append(observation2[2][49])
        self.newstate[5].append(observation2[2][49])
        self.newstate[6].append(observation2[3][49])
        self.newstate[7].append(observation2[3][49])
        self.newstate[8].append(observation2[4][49])
        self.newstate[9].append(observation2[5][49])
        self.newstate[10].append(observation2[6][49])
        self.newstate[11].append(observation2[6][49])
        self.newstate[12].append(observation2[7][49])
        self.newstate[13].append(observation2[7][49])
        # self.newstate[14].append(pre_state[8][49] / 2)
        # self.newstate[15].append(pre_state[8][49] / 2)
        # self.newstate[16].append(pre_state[9][49] / 2)
        # self.newstate[17].append(pre_state[9][49] / 2)
        # self.newstate[18].append(pre_state[10][49] / 2)
        # self.newstate[19].append(pre_state[10][49] / 2)

        return self.newstate

    def set_newstate(self, pre_state, observation2): # reset 할 때만 사용
        self.newstate = [[], [], [], [], [], [], [], [], [], [], [], [], [], []]
        for i in range(50):
            self.newstate[0].append(observation2[0][i])
            self.newstate[1].append(observation2[0][i])
            self.newstate[2].append(observation2[1][i])
            self.newstate[3].append(observation2[1][i])
            self.newstate[4].append(observation2[2][i])
            self.newstate[5].append(observation2[2][i])
            self.newstate[6].append(observation2[3][i])
            self.newstate[7].append(observation2[3][i])
            self.newstate[8].append(observation2[4][i])
            self.newstate[9].append(observation2[5][i])
            self.newstate[10].append(observation2[6][i])
            self.newstate[11].append(observation2[6][i])
            self.newstate[12].append(observation2[7][i])
            self.newstate[13].append(observation2[7][i])
            # self.newstate[14].append(pre_state[8][i] / 2)
            # self.newstate[15].append(pre_state[8][i] / 2)
            # self.newstate[16].append(pre_state[9][i] / 2)
            # self.newstate[17].append(pre_state[9][i] / 2)
            # self.newstate[18].append(pre_state[10][i] / 2)
            # self.newstate[19].append(pre_state[10][i] / 2)


def main():
    # env = gym.make(ENV_NAME)
    global accuracy, last_observation



    if TRAIN:  # Train mode
        env = Env()
        agent = Agent(10)
        # 4*20*50 만들 때, 빈 index 만들고 env.step(0)으로 observation 갖고와서 빈 index에 append후 reshape & stack 해야될 것 같음.
        for episode in range(NUM_EPISODES):
            terminal = False
            observation, observation2 = env.reset() # 11*50
            state2 = observation # for loop에만 있으면 지역 변수로 없어져서 선언
            # print(np.shape(observation))

            agent.set_newstate(observation, observation2)
            state = []
            for _ in range(4):
                observation, observation2, _, _, _ = env.step(0)
                state2 = observation #실제 값
                observation2 = agent.make_newstate(observation, observation2)
                observation2 = np.reshape(observation2, (FRAME_WIDTH, FRAME_HEIGHT, 1)) # 20*50*1
                # print(observation2)
                state.append(observation2)
            state = np.swapaxes(state, 0, 3) # 1*14*50*4
            # print(np.shape(observation2))
            # print(np.shape(state))
            # print(state)

            while not terminal:
                action = agent.get_action(state, state2)
                observation, observation2, reward, terminal, accuracy = env.step(action)
                state2 = observation
                # print(state2)
                observation2 = agent.make_newstate(observation, observation2)

                # env.render()
                observation2 = np.reshape(observation2, (1, FRAME_WIDTH, FRAME_HEIGHT, 1))

                # print(np.shape(state))
                # print(np.shape(observation))
                state = agent.run(state, action, reward, terminal, observation2)


    else:  # Test mode
        env = Env(Train=False, maxtime=100)
        agent = Agent(10)
        # env.monitor.start(ENV_NAME + '-test')

        for episode in range(NUM_EPISODES_AT_TEST):
            real = []
            pred = []
            count = 0
            terminal = False
            observation, observation2 = env.reset() # 11*50
            state2 = observation  # for loop에만 있으면 지역 변수로 없어져서 선언
            agent.set_newstate(observation, observation2)
            state = []
            pos_reward = 0
            neg_reward = 0

            for _ in range(4):
                observation, observation2, _, _, _ = env.step(0)
                real.append(observation[3][49])
                pred.append(None)
                state2 = observation  # 20*50
                observation2 = agent.make_newstate(observation, observation2)
                # print(np.shape(observation))
                observation2 = np.reshape(observation2, (FRAME_WIDTH, FRAME_HEIGHT, 1))  # 20*50*1
                # print(np.shape(observation))
                state.append(observation2)
            state = np.swapaxes(state, 0, 3)  # 1*14*50*4
            while not terminal:
                action = agent.get_action_at_test(state, state2)
                # print(action)
                observation,observation2, reward, terminal, accuracy = env.step(action)
                real.append(observation[3][49])
                if reward >= 0:
                    pos_reward = pos_reward + reward
                    pred.append(observation[3][49])
                    count = count + 1
                else:
                    neg_reward = neg_reward + reward
                    pred.append(None)
                state2 = observation
                # print(observation[8][49])
                # print(observation[9][49])
                # print(observation[10][49])
                observation2 = agent.make_newstate(observation, observation2)
                observation2 = np.reshape(observation2, (1, FRAME_WIDTH, FRAME_HEIGHT, 1))
                state = np.append(state[:, :, :, 1:], observation2, axis=3)
            print("pos reward = ", pos_reward)
            print("neg reward = ", neg_reward / (100 - count))
            plt.figure()
            plt.plot(pred, 'ro')
            plt.plot(real, 'b', label='Actual Close')
            plt.legend()
            plt.savefig('./%d.png' %env.episode)
            print(count)
    agent.f.close()

if __name__ == '__main__':
    main()