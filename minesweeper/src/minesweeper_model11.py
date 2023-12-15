import numpy as np
from tensorflow import keras
from minesweeper import Minesweeper
from minesweeper_model import MinesweeperModel


class MinesweeperModel11(MinesweeperModel):
    input_width = 11

    def __init__(self, model_path, rx, ry):
        self.rx = rx
        self.ry = ry
        super().__init__(model_path)

    def actions_to_inputs(self, action_set, state, solution):
        # 0, 1, 2, 3, 4, 5, 6, 7, 8, unknown, oob
        samples = np.zeros((len(action_set), self.rx * self.ry * MinesweeperModel11.input_width))
        for i, action in enumerate(action_set):
            x, y = action
            for dx, dy in np.ndindex((self.rx, self.ry)):
                px = x + dx - self.rx // 2
                py = y + dy - self.ry // 2
                data_offset = (dx * self.rx * MinesweeperModel11.input_width) + dy * MinesweeperModel11.input_width

                if not Minesweeper.in_bounds(px, py, state):
                    samples[i, data_offset + 10] = 1
                    continue

                if not state[px, py]:
                    samples[i, data_offset + 9] = 1
                    continue

                index = solution[px, py]
                samples[i, data_offset + index] = 1

        return samples

    def create_model(self):
        model = keras.Sequential([
            keras.layers.Input(shape=(MinesweeperModel11.input_width * self.rx * self.ry,)),
            keras.layers.Dense(MinesweeperModel11.input_width * self.rx * self.ry, activation='relu'),
            keras.layers.Dense(MinesweeperModel11.input_width * self.rx * self.ry, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
