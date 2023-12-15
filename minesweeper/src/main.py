import pickle
import numpy as np
import json
import os
import random
import time
from game_data import GameData
from minesweeper import Minesweeper
from minesweeper_model9 import MinesweeperModel9
from minesweeper_model11 import MinesweeperModel11

current_time = int(time.time())
np.random.seed(current_time)

def test_model(model, num_games=1000, game_parameters=None, verbose=False):
    if game_parameters is None:
        game_parameters = [[10, 10, 10], [16, 16, 40], [36, 16, 99]]

    results = []
    counter = 0
    for parameters in game_parameters:
        w, h, b = parameters
        win_counter = 0
        for _ in range(num_games):
            game = Minesweeper(w, h, b)

            # Start the game
            x, y = random.choice(np.argwhere(game.bombs == False))
            game.select(x, y)

            while True:
                x, y = model.select_action(game)
                game.select(x, y)

                if game.has_lost():
                    x, y = model.select_action(game)
                    game.select(x, y)
                    break

                if game.has_won():
                    win_counter += 1
                    break

            if verbose:
                counter += 1
                print(
                    f"Completed {counter} of {num_games * len(game_parameters)}, win rate {win_counter / float(counter)}")
        results.append(win_counter / float(num_games))

    return results


def load_training_data(file_path='./training_data/training_data.pkl'):
    # Load existing training data if file exists
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            training_data = pickle.load(file)
    else:
        training_data = []

    return training_data


def save_training_data(training_data, file_path='./training_data/training_data.pkl'):
    # Save the updated training data
    with open(file_path, 'wb') as file:
        pickle.dump(training_data, file)


def generate_training_data(num_games=1000, game_parameters=None, verbose=False):
    if game_parameters is None:
        game_parameters = [[10, 10, 10], [16, 16, 40], [36, 16, 99]]

    training_data = []

    counter = 0
    for parameters in game_parameters:
        w, h, b = parameters
        for _ in range(num_games):
            game = Minesweeper(w, h, b)
            x, y = random.choice(np.argwhere(game.bombs == False))
            game.select(x, y)
            game_data = GameData(game.bombs, game.solution)
            while True:
                game_data.add_step(game.get_actions(), game.state)
                x, y = random.choice(np.argwhere(game.bombs == False))
                game.select(x, y)
                if game.has_won():
                    break
                assert not game.has_lost()

            training_data.append(game_data)

            if verbose:
                counter += 1
                print(f"Game {counter} generated of {num_games * len(game_parameters)}.")

    return training_data


training_data = load_training_data()
new_training_data = generate_training_data(20, verbose=True)
training_data = training_data + new_training_data
save_training_data(training_data)

model11_3 = MinesweeperModel11('./models/minesweeper_models11_3.keras', 3, 3)
model11_3.train(new_training_data, verbose=True)
model11_3.save_models()
results11_3 = test_model(model11_3, 25, verbose=True)

model9 = MinesweeperModel9('./models/minesweeper_models9.keras', 5, 5)
model9.train(new_training_data, verbose=True)
model9.save_models()
results9 = test_model(model9, 25, verbose=True)


model11_5 = MinesweeperModel11('./models/minesweeper_models11_5.keras', 5, 5)
model11_5.train(new_training_data, verbose=True)
model11_5.save_models()
results11_5 = test_model(model11_3, 25, verbose=True)

model11_9 = MinesweeperModel11('./models/minesweeper_models11_9.keras', 9, 9)
model11_9.train(new_training_data, verbose=True)
model11_9.save_models()
results11_9 = test_model(model11_9, 25, verbose=True)

with open('results.json', 'w') as file:
    json.dump({
        "results9": results9,
    }, file)
