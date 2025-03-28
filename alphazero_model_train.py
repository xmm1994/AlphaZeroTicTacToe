import numpy as np
import tensorflow as tf
from tensorflow import keras
import random, copy
from tools import *

# 使用 tf.keras.utils.register_keras_serializable 注册自定义模型类
@tf.keras.utils.register_keras_serializable()
class AlphaZeroModel(keras.Model):
    def __init__(self):
        super(AlphaZeroModel, self).__init__()
        # 使用 Sequential 模型构建主体部分
        self.model = keras.Sequential([
            keras.layers.InputLayer(shape=(10,), dtype=tf.float32),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu')
        ])
        # 策略输出层
        self.policy_output = keras.layers.Dense(9, activation='softmax')
        # 价值输出层
        self.value_output = keras.layers.Dense(1, activation='tanh')

    def call(self, inputs):
        # 确保输入为 float32 类型的张量
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        inputs = tf.cast(inputs, dtype=tf.float32)
        # 通过 Sequential 模型进行前向传播
        x = self.model(inputs)
        # 得到策略输出
        policy = self.policy_output(x)
        # 得到价值输出
        value = self.value_output(x)
        return policy, value

    def get_config(self):
        # 返回模型的配置信息
        config = super(AlphaZeroModel, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        # 根据配置信息重新创建模型实例
        return cls()

# 定义 MCTS 节点类
class MCTSNode:
    def __init__(self, board, prob = 1.0, parent=None):
        # 当前节点对应的棋盘状态
        self.board = board
        # 父节点
        self.parent = parent
        # 子节点字典，键为落子位置，值为对应的子节点
        self.children = {}
        # 该节点的访问次数
        self.visits = 0
        # 该节点的状态价值
        self.Q = 0
        # 未尝试的落子位置列表
        self.untried_actions = [i for i, cell in enumerate(board) if cell == '']
        # 存储神经网络预测的动作概率
        self.P = prob

    def is_fully_expanded(self):
        # 判断该节点是否已经完全扩展，即是否还有未尝试的落子位置
        return len(self.untried_actions) == 0
    
    def select_child(self, c_puct=1.4):
        best_score = -float('inf')
        best_child = None
        for action, child in self.children.items():
            U = c_puct * child.P * np.sqrt(self.visits) / (1 + child.visits)
            score = child.Q + U
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, action_probs):
        # 扩展一个未尝试的落子位置，创建一个新的子节点
        for action, prob in action_probs:
            board = self.board.copy()
            board[action] = self.board[-1]
            board[-1] = 'O' if self.board[-1] == 'X' else 'X'
            self.children[action] = MCTSNode(board, prob, self)

    def simulate(self, model):
        # 使用神经网络模拟当前局面，得到策略和价值
        state = np.array([1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in self.state]).reshape(1, -1).astype(np.float32)
        policy, value = model(state)
        self.policy = policy.numpy()[0]
        return value.numpy()[0][0]

    def backpropagate(self, state_value):
        # 反向传播模拟结果，更新节点的访问次数和累计胜利次数
        self.visits += 1
        self.Q += 1.0*(state_value - self.Q) / self.visits
        if self.parent:
            # 递归更新父节点
            self.parent.backpropagate(-state_value)
            

class AlphaZeroPlayer:
    def __init__(self, model = None, optimize = None, policy_loss_fn = None, value_loss_fn = None, num_games = 1000, simulations = 1000):
        self.model = model
        self.policy_loss_fn = policy_loss_fn
        self.value_loss_fn = value_loss_fn
        self.num_games = num_games
        self.simulations = simulations
        self.train_data = []
    
    def playout(self, node):
        while(1):
            if node.children == {}:
                # print("is_leaf, board: ", node.board)
                break
            node = node.select_child()
        
        state = np.array([1.0 if cell == 'X' else -1 if cell == 'O' else 0.0 for cell in node.board])
        state = np.expand_dims(state, axis=0)
        pred_probs, pred_state_value = self.model(state)
        probs = pred_probs.numpy()[0]
        state_value = pred_state_value.numpy()[0][0]
        # print('probs: ', probs)
        # print('state_value: ', state_value)
        availables_action_probs = [(i, probs[i]) for i in range(len(node.board) - 1) if node.board[i] == '']
        # print('availables_action_probs: ', availables_action_probs)
        
        win = check_winner(node.board)
        # print("win: ", win)
        if win is None:
            node.expand(availables_action_probs)
        elif win == 'tie':
            state_value = 0.0
            # print('tie board: ', node.board)
        elif win == node.board[-1]:
            state_value = -1.0
            # print('loss board: ', node.board)
        else:
            state_value = 1.0
            # print('win board: ', node.board)
        
        node.backpropagate(state_value)

    def softmax(self, x):
        probs = np.exp(x - np.max(x))
        probs /= np.sum(probs)
        return probs

    def get_action(self, node = None, is_selfplay = False, board = None):
        if is_selfplay == False:
            board.append('O') # 添加当前玩家
            node = MCTSNode(board)
        # 返回最优action以及所有action的概率
        for i in range(self.simulations):
            self.playout(node)
        
        act_visits = [(action, child.visits) for action, child in node.children.items()]
        actions, visits = zip(*act_visits)
        act_probs = self.softmax(np.log(np.array(visits) + 1e-10))
            
        if is_selfplay:
            move = np.random.choice(
                actions,
                p=0.75*act_probs + 0.25*np.random.dirichlet(0.3*np.ones(len(act_probs)))
            )
        else:
            # play 阶段选择概率最大的action
            print("act_visits: ", act_visits)
            move = max(act_visits, key=lambda x: x[1])[0]
        
        # 不合法action概率设为0
        res_probs = [0.0] * 9
        for i in range(len(actions)):
            res_probs[actions[i]] = act_probs[i]
        
        if is_selfplay:
            return move, res_probs
        else:
            return move

    # 自对弈生成训练数据
    def collect_self_play_data(self):
        states = []
        probs = []
        values = []
        board = [''] * 9
        current_player = 'X'
        board.append(current_player) # 最后一位表示当前player
        game_history = []

        root = MCTSNode(board)
        while True:
            winner = check_winner(root.board)
            if winner is None:
                move, prob = self.get_action(node = root, is_selfplay = True)
                states.append(np.array([1.0 if cell == 'X' else -1 if cell == 'O' else 0.0 for cell in root.board]))
                probs.append(prob)
                game_history.append((root.board.copy(), root.board[-1]))
                root = root.children[move]
            else:
                states.append(np.array([1.0 if cell == 'X' else -1 if cell == 'O' else 0.0 for cell in root.board]))
                probs.append([0.0] * 9)
                game_history.append((root.board.copy(), root.board[-1]))
                for _, player in game_history:
                    if winner == 'tie':
                        values.append(0.0)
                    elif winner == player:
                        values.append(-1.0)
                    else:
                        values.append(1.0)
                return np.array(states), np.array(probs), np.array(values)

    # 训练模型
    def train_model(self):
        for game in range(self.num_games):
            states, policies, values = self.collect_self_play_data()
            self.train_data.append([states, policies, values])
            with tf.GradientTape() as tape:
                policy_preds, value_preds = self.model(states)
                policy_loss = self.policy_loss_fn(policies, policy_preds)
                value_loss = self.value_loss_fn(values, value_preds)
                total_loss = policy_loss + value_loss

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if game % 100 == 0:
                print(f"Game {game}, Loss: {total_loss.numpy()}, policy_loss: {policy_loss.numpy()}, value_loss: {value_loss.numpy()}")
                states = np.array([[0.0]*9 + [-1.0]])
                policy_preds, value_preds = self.model(states)
                print(policy_preds.numpy()[0], value_preds.numpy()[0][0])
    

if __name__ == "__main__":
    # 初始化 AlphaZero 模型
    model = AlphaZeroModel()
    # 定义优化器，使用 Adam 优化器，学习率为 0.001
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    # 定义策略损失函数，使用交叉熵损失函数
    policy_loss_fn = keras.losses.CategoricalCrossentropy()
    # 定义价值损失函数，使用均方误差损失函数
    value_loss_fn = keras.losses.MeanSquaredError()

    alphazero_player = AlphaZeroPlayer(model, optimizer, policy_loss_fn, value_loss_fn, 1000, 100)

    alphazero_player.train_model()
    # 保存训练好的模型，添加 .keras 扩展名
    alphazero_player.model.save('./model/alphazero_tictactoe_model.keras')
    
    with open('alphazero_train_data.txt', 'w', encoding='utf-8') as file:
        for x in alphazero_player.train_data:
            file.write(str(x)+'\n')