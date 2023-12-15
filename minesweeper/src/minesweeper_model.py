import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from src.minesweeper import Minesweeper
import random


class MinesweeperModel:

    def __init__(self, model_path):
        self.load_or_create_model(model_path)

    def load_or_create_model(self, model_path):
        self.model_path = model_path
        if os.path.exists(self.model_path):
            self.model = load_model(model_path)
        else:
            self.model = self.create_model()

    def actions_to_inputs(self, action_set, state, solution):
        raise NotImplemented()

    def create_model(self):
        raise NotImplemented()

    def train(self, training_data, verbose=False):

        pairings = [(i, x) for i, game in enumerate(training_data) for x in range(len(game.actions))]
        random.shuffle(pairings)
        for data_idx, step_idx in pairings:
            data = training_data[data_idx]
            actions, state, bombs, solution = (data.actions[step_idx], data.states[step_idx],
                                               data.bombs, data.solution)
            inputs = self.actions_to_inputs(actions, state, solution)
            X, Y = zip(*actions)
            outputs = 1.0 - bombs[X, Y].astype(np.float64)
            outputs = outputs.reshape((*outputs.shape, 1))
            self.model.train_on_batch(inputs, outputs)


    def predict(self, inputs):
        return self.model.predict(inputs, verbose=None)

    def save_models(self):
        self.model.save(self.model_path)

    def select_action(self, game):
        actions = game.get_actions()
        inputs = self.actions_to_inputs(actions, game.state, game.solution)
        predictions = self.model.predict(inputs, verbose=None)
        idx = np.argmax(predictions)
        return actions[idx]
