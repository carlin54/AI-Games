import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model

from src.minesweeper import Minesweeper
from src.minesweeper_model import MinesweeperModel

class MinesweeperModel9(MinesweeperModel):
    input_width = 9
    def __init__(self, model_path, rx, ry):
        self.rx = rx
        self.ry = ry
        super().__init__(model_path)

    def actions_to_inputs(self, actions, state, solution):
        inputs = np.zeros((len(actions), self.rx * self.ry * MinesweeperModel9.input_width))
        for i, action in enumerate(actions):
            x, y = action
            for dx, dy in np.ndindex((self.rx, self.ry)):
                px = x + dx - self.rx // 2
                py = y + dy - self.ry // 2
                data_offset = (dx * self.rx * MinesweeperModel9.input_width) + dy * MinesweeperModel9.input_width

                if not Minesweeper.in_bounds(px, py, state):
                    continue

                index = solution[px, py]
                inputs[i, data_offset + index] = 1

        return inputs

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(MinesweeperModel9.input_width * self.rx * self.ry,)),
            keras.layers.Dense(MinesweeperModel9.input_width * self.rx * self.ry, activation='relu'),
            keras.layers.Dense(MinesweeperModel9.input_width * self.rx * self.ry, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


