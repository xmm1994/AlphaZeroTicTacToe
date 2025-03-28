import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
import argparse
from alphazero_model_train import AlphaZeroPlayer
from mcts_player import MCTSPlayer

app = Flask(__name__)

parser = argparse.ArgumentParser(description='tictactoe')
parser.add_argument('player_name', help='player name')
args = parser.parse_args()
if args.player_name == 'AlphaZero':
    model = keras.models.load_model('./model/alphazero_tictactoe_model.keras')
    player = AlphaZeroPlayer(model = model, simulations = 100)
elif args.player_name == 'MCTS':
    player = MCTSPlayer(simulations = 100)
else:
    print("player name error, please input AlphaZero or MCTS")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def get_move():
    data = request.get_json()
    board = data['board']
    move = player.get_action(board = board)
        
    return jsonify({'move': move})

if __name__ == '__main__':
    app.run(debug=True)