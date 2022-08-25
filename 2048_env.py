import gym
import random
import numpy as np
import tensorflow as tf
import math as ma
from collections import deque
from gym import spaces

class Logic2048:
    def __init__(self, matrix_shape=(4, 4)):
        self.matrix = [[0 for i in range(matrix_shape[0])] for i in range(matrix_shape[1])]
        self.generate_new_num()
        self.generate_new_num()

    def zero_to_end(self, row):
        # 倒着遍历
        for i in range(len(row) - 1, -1, -1):
            if row[i] == 0:
                row.pop(i)
                row.append(0)

    def merge(self, row):
        # 移动并合并
        self.zero_to_end(row)
        for i in range(len(row) - 1):
            if row[i] == 0:
                break
            if row[i] == row[i + 1]:
                row[i] *= 2
                row.pop(i + 1)
                row.append(0)

    def left(self):
        for row in self.matrix:
            self.merge(row)

    def right(self):
        for row in self.matrix:
            row.reverse()
            self.merge(row)
            row.reverse()

    def matrix_transpose(self):
        # 矩阵转置
        for col_index in range(1, len(self.matrix)):
            for row_index in range(col_index, len(self.matrix)):
                self.matrix[row_index][col_index - 1], self.matrix[
                    col_index - 1][row_index] = self.matrix[
                        col_index -
                        1][row_index], self.matrix[row_index][col_index - 1]

    def up(self):
        self.matrix_transpose()
        self.left()
        self.matrix_transpose()

    def down(self):
        self.matrix_transpose()
        self.right()
        self.matrix_transpose()

    def move(self, direction):
        dir_dict = {
            'left': self.left,
            'right': self.right,
            'up': self.up,
            'down': self.down
        }
        func = dir_dict.get(direction)
        if func:
            func()

    def get_empty_position(self):
        empty_position = []
        for row_index in range(len(self.matrix)):
            for col_index in range(len(self.matrix[row_index])):
                if self.matrix[row_index][col_index] == 0:
                    empty_position.append((row_index, col_index))
        return empty_position

    def generate_new_num(self):
        self.empty_position = self.get_empty_position()
        if not self.empty_position:
            return False
        row_index, col_index = random.choice(self.empty_position)
        self.matrix[row_index][col_index] = 4 if random.randint(1, 10) == 1 else 2
        self.empty_position.remove((row_index, col_index))
        return True

    def is_game_win(self, flag):
        for r in self.matrix:
            if flag in r:
                return True
        return False

    def is_game_over(self):
        for r in range(len(self.matrix)):
            for c in range(len(self.matrix[r]) - 1):
                # 判断是否还可以走
                if self.empty_position:
                    return False
                # 判断是否还可以相加
                if self.matrix[r][c] == self.matrix[r][c + 1] or self.matrix[c][r] == self.matrix[c + 1][r]:
                    return False
        return True

    def calc_reward(self, step = 0, max_step = 3000):
        r = 0
        buff = 0
        for i in self.matrix:
            for j in i:

                if j == 128:
                    buff += 0.25
                if j == 256:
                    buff += 0.55
                if j == 512:
                    buff += 1.15
                if j == 1024:
                    buff += 2.35
                if j == 2048:
                    buff += 4.75
                if j == 4096:
                    buff += 9.55

                r += j
        
        r /= 16
        r *= (1 + buff / 100)
        r *= (1 - 0.15 * (max_step - step))
        return r

    def get_flatten_state(self):
        s = []
        for i in self.matrix:
            for j in i:
                s.append(j)
        return np.array(s)

    def move_able(self, di):
        now = self.matrix.copy()
        self.move(di)
        if now == self.matrix:
            return False
        else:
            self.matrix = now
            return True

    def get_max(self):
        m = -114514
        for i in self.matrix:
            for j in i:
                if j >= m:
                    m = j
        return m

class game2048(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(low = 0, high = 65536, shape = (16, )) # 16 grids
        self.game = Logic2048()

    def step(self, action):
        if action == 0:
            ac = "up"
        if action == 1:
            ac = "down"
        if action == 2:
            ac = "left"
        if action == 3:
            ac = "right"
        self.game.move(ac)
        self.game.generate_new_num()

        state = self.game.get_flatten_state()
        reward = self.game.calc_reward()
        done = self.game.is_game_over()
        info = {"move_able": self.game.move_able(ac)}
        return state, reward, done, info
    
    def reset(self):
        self.game = Logic2048()
        state = self.game.get_flatten_state()
        return state
    
    def render(self, mode='human'):
        print("----------")
        print(self.game.get_flatten_state())
        print("rewards: " + str(self.game.calc_reward()))

    def seed(self, seed = None):
        pass

class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units = 16, activation = tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units = 16, activation = tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units = 16, activation = tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(units = 4)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x

    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)

def tran(l):
    return [ma.log2(t + 1) / 16 for t in l]

if __name__ == "__main__":

    num_episodes = 10000              # 游戏训练的总episode数量
    num_exploration_episodes = 500  # 探索过程所占的episode数量
    max_len_episode = 3000          # 每个episode的最大回合数
    batch_size = 128                # 批次大小
    learning_rate = 1e-3            # 学习率
    gamma = 1.                      # 折扣因子
    initial_epsilon = 1.            # 探索起始时的探索率
    final_epsilon = 0.05            # 探索终止时的探索率

    env = game2048()

    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen = 10000) # 使用一个 deque 作为 Q Learning 的经验回放池
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):
        state = env.reset()             # 初始化环境，获得初始状态
        epsilon = max(initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes, final_epsilon)  # 当前探索率
        for t in range(max_len_episode):
            # env.render()                                # 对当前帧进行渲染
            if random.random() < epsilon:               # epsilon-greedy 探索策略，以 epsilon 的概率选择随机动作
                action = env.action_space.sample()      # 选择随机动作（探索）
                # print("action: Random")
            else:
                state_ = tran(state)
                action = model.predict(np.expand_dims(np.array(state_), axis = 0)).numpy()   # 选择模型计算出的 Q Value 最大的动作
                action = action[0]
                # print("action: " + str(action))


            # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
            next_state, reward, done, info = env.step(action)
            next_state_ = tran(next_state)
            state_ = tran(state)
            # 如果游戏Game Over，给予负奖励
            if done:
                reward = -10
            # if info["move_able"] == False:
                # reward = -10
                # done = True
            # 将(state, action, reward, next_state)的四元组（外加 done 标签表示是否结束）放入经验回放池
            replay_buffer.append((state_, action, reward, next_state_, 1 if done else 0))
            # 更新当前 state
            state = next_state

            if done:                                    # 游戏结束则退出本轮循环，进行下一个 episode
                print("episode %4d, epsilon %.4f, score %d, max %d" % (episode_id, epsilon, t, env.game.get_max()))
                break

        if len(replay_buffer) >= batch_size:
            # 从经验回放池中随机取一个批次的四元组，并分别转换为 NumPy 数组
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                map(np.array, zip(*random.sample(replay_buffer, batch_size)))

            q_value = model(batch_next_state)
            y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)  # 计算 y 值
            with tf.GradientTape() as tape:
                loss = tf.keras.losses.mean_squared_error(  # 最小化 y 和 Q-value 的距离
                    y_true=y,
                    y_pred=tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=4), axis=1)
                )
                # print(y)
                # print(tf.reduce_sum(model(batch_state) * tf.one_hot(batch_action, depth=4), axis=1))
                # print(loss)
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))       # 计算梯度并更新参数
