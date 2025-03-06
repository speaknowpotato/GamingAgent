import numpy as np
from logic import move, checkGameStatus, fillTwoOrFour, get_last_score


class Game2048Env:
    def __init__(self):
        self.board = None
        self.score = 0
        self.reset()

    def reset(self):
        """重置环境"""
        self.board = fillTwoOrFour([[0] * 4 for _ in range(4)], iter=2)
        self.score = 0
        return self._get_state()

    def _get_state(self):
        """将棋盘转换为神经网络输入"""
        # 使用对数尺度并归一化
        state = np.array(self.board, dtype=np.float32)
        np.log2(state, out=state, where=state > 0)
        return state / 16.0  # 最大值为16 (2^16=65536)

    def step(self, action):
        """
        执行动作并返回新的状态、奖励、是否终止
        action: 0-3对应上、右、下、左
        """
        action_map = ["w", "d", "s", "a"]
        old_board = [row[:] for row in self.board]
        self.board = move(action_map[action], self.board)

        # 计算奖励
        reward = 0
        done = False

        # 基础奖励：新合并的分数
        new_score = get_last_score()
        reward += new_score * 0.1

        # 启发式奖励
        empty_cells = sum(row.count(0) for row in self.board)
        reward += empty_cells * 0.5  # 鼓励保留空位

        max_tile = max(max(row) for row in self.board)
        reward += max_tile * 0.001  # 鼓励增大最大方块

        # 检查游戏状态
        status = checkGameStatus(self.board)
        if status != "PLAY":
            done = True
            if status == "WIN":
                reward += 1000
            else:
                reward -= 500

        # 如果棋盘没有变化，给予惩罚
        if self.board == old_board:
            reward -= 10

        return self._get_state(), reward, done, {}
