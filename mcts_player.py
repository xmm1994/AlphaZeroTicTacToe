import random
import copy
import math

class MCTSNode:
    def __init__(self, state, parent=None):
        # 当前节点对应的棋盘状态
        self.state = state
        # 父节点
        self.parent = parent
        # 子节点字典，键为落子位置，值为对应的 MCTSNode 对象
        self.children = {}
        # 该节点被访问的次数
        self.visits = 0
        # 该节点对应的模拟中获胜的次数
        self.wins = 0
        # 未尝试过的落子位置列表
        self.untried_actions = [i for i, cell in enumerate(state) if cell == '']

    def is_fully_expanded(self):
        # 判断该节点是否已经完全扩展，即是否还有未尝试的落子位置
        return len(self.untried_actions) == 0

    def select_child(self, c=1.4):
        # 根据 UCT（Upper Confidence Bound applied to Trees）公式选择一个子节点
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            # UCT 公式： exploitation + exploration
            score = child.wins / child.visits + c * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self):
        # 扩展一个新的子节点
        action = self.untried_actions.pop()
        new_state = self.state.copy()
        # 根据当前局面判断轮到哪个玩家落子
        new_state[action] = 'X' if self.state.count('X') <= self.state.count('O') else 'O'
        child = MCTSNode(new_state, self)
        self.children[action] = child
        return child

    def simulate(self):
        # 模拟一局游戏直到结束
        current_state = self.state.copy()
        current_player = 'X' if self.state.count('X') > self.state.count('O') else 'O'
        next_player = 'O' if self.state.count('X') <= self.state.count('O') else 'O'
        while True:
            winner = check_winner(current_state)
            if winner == 'tie':
                return 0
            elif winner == current_player:
                return 1
            elif winner != 'none':
                return -1
            available_moves = [i for i, cell in enumerate(current_state) if cell == '']
            move = random.choice(available_moves)
            current_state[move] = next_player
            next_player = 'O' if next_player == 'X' else 'X'

    def backpropagate(self, result):
        # 反向传播模拟结果，更新节点的访问次数和获胜次数
        self.visits += 1
        self.wins += result
        if self.parent:
            # 递归更新父节点，注意结果取反
            self.parent.backpropagate(0 - result)

class MCTSPlayer:
    def __init__(self, simulations=100):
        self.simulations = simulations
        
    def get_action(self, board):
        # 蒙特卡洛树搜索主函数
        root = MCTSNode(board)
        for _ in range(self.simulations):
            node = root
            # 选择阶段：从根节点开始，根据 UCT 公式选择子节点直到叶子节点
            while node.is_fully_expanded() and node.children:
                node = node.select_child()
            # 扩展阶段：如果叶子节点还有未尝试的动作，则扩展一个新的子节点
            if not node.is_fully_expanded():
                node = node.expand()
            # 模拟阶段：从新节点开始模拟一局随机游戏
            result = node.simulate()
            # 反向传播阶段：将模拟结果反向传播更新节点信息
            node.backpropagate(result)
        # 选择访问次数最多的子节点对应的动作作为最佳动作
        # best_move = max(root.children, key=lambda k: root.children[k].wins / root.children[k].visits)
        best_move = max(root.children, key=lambda k: root.children[k].visits)
        for k, v in root.children.items():
            print(k, v.visits, v.wins)
        return best_move