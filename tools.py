def check_winner(board):
    # 检查当前棋盘状态是否有玩家获胜或平局
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
        [0, 4, 8], [2, 4, 6]  # 对角线
    ]
    for combination in winning_combinations:
        a, b, c = combination
        if board[a] and board[a] == board[b] and board[a] == board[c]:
            return board[a]
    if '' not in board:
        return 'tie'
    return None