import math

import keras as keras
import numpy as np
from numpy import random
import tensorflow as tf
from enum import Enum


class Status:
    @staticmethod
    def WIN():
        return 1

    @staticmethod
    def LOST():
        return -1

    @staticmethod
    def NILL():
        return 0


class Move(Enum):
    UP = dict([("start", (0, 3)), ("dir", (0, -1)),
               ("expected", np.array([1.0, 0.0, 0.0, 0.0]).reshape((1, 4))),
               ("mask", np.array([0.0, 1.0, 1.0, 1.0]).reshape((1, 4)))])
    DOWN = dict([("start", (0, 0)), ("dir", (0, 1)),
                 ("expected", np.array([0.0, 1.0, 0.0, 0.0]).reshape((1, 4))),
                ("mask", np.array([1.0, 0.0, 1.0, 1.0]).reshape((1, 4)))])
    LEFT = dict([("start", (3, 0)), ("dir", (-1, 0)),
                 ("expected", np.array([0.0, 0.0, 1.0, 0.0]).reshape((1, 4))),
                 ("mask", np.array([1.0, 1.0, 0.0, 1.0]).reshape((1, 4)))])
    RIGHT = dict([("start", (0, 0)), ("dir", (1, 0)),
                  ("expected", np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))),
                  ("mask", np.array([1.0, 1.0, 1.0, 0.0]).reshape((1, 4)))])

class Game2048:


    board = np.zeros(shape=(4, 4), dtype=np.int32)
    end = 4
    state_max = 2048


    def total_score(self):
        return 0

    def new_game(self):
        self.board = np.zeros(shape=(4, 4), dtype=np.int32)
        self.add_cell()
        self.add_cell()

    def out_of_bounds(self, x) -> bool:
        return x > 3 or x < 0

    def in_bounds(self, x) -> bool:
        return 3 >= x >= 0

    def get(self, x, y):
        return self.board[y-3, x]

    def set(self, x, y, val):
        self.board[y-3, x] = val

    def add_cell(self):
        empty_cells = []
        for x in range(0, 4):
            for y in range(0, 4):
                val = self.get(x, y)
                if val == 0:
                    empty_cells.append((x, y))
        n = len(empty_cells)
        if n > 0:
            sel = random.randint(n)
            x, y = empty_cells.pop(sel)
            self.set(x, y, 2)
            return True
        else:
            return False

    def step(self, move):

        # moves cursor across row
        mx, my = move.value['dir']

        # moves down the column
        iy, ix = abs(mx), abs(my)

        # cursor start
        sx, sy = move.value['start']

        x = sx
        y = sy

        score = 0
        while self.in_bounds(sx) and self.in_bounds(sy):
            row = self.get_row(sx, sy, mx, my)
            [compressed_row, gained_score] = self.compress_row(row)
            self.set_row(sx, sy, mx, my, compressed_row)
            [sx, sy] = [sx + ix, sy + iy]
            score = score + gained_score

        success = self.add_cell()
        if self.has_lost():
            return score, Status.LOST
        else:
            return score, self.has_won()


    def has_won(self):
        for x in range(0, 4):
            for y in range(0, 4):
                val = self.get(x, y)
                if val >= self.end:
                    return Status.WIN

        return Status.NILL

    def print_board(self):
        for y in range(0, 4):
            for x in range(0, 4):
                print(str(self.get(x, y)) + "\t", end="")
            print()

    def get_row(self, sx, sy, mx, my):
        buffer = []
        x = sx
        y = sy
        while self.in_bounds(x) and self.in_bounds(y):
            val = self.get(x, y)
            buffer.append(val)
            x = x + mx
            y = y + my

        return buffer

    def set_row(self, sx, sy, mx, my, buffer):
        x = sx
        y = sy
        val = None
        while self.in_bounds(x) and self.in_bounds(y):
            val = buffer.pop(0) if len(buffer) > 0 else 0
            self.set(x, y, val)
            x = x + mx
            y = y + my

        return buffer

    def get_board_state(self):
        state = np.zeros(shape=(1, self.get_state_size()))
        len = int(math.log2(self.state_max))

        for x in range(0, 4):
            for y in range(0, 4):
                cursor = ((x * 4) + y) * len
                val = self.get(x, y)
                idx = 0
                if val > 0:
                    idx = int(math.log2(val))
                idx = cursor + idx
                state[0, idx] = 1

        return state

    def get_state_size(self):
        return int(math.log2(self.state_max) * 4 * 4)

    @staticmethod
    def compress_row(row):
        carry = val = None
        compressed_row = []
        space = []
        score = 0
        row = row[::-1]
        for i in range(0, len(row)):
            if row[i] == 0:
                space.append(0)
                continue

            if carry is None:
                carry = row[i]
                continue
            else:
                val = row[i]

            if carry == val:
                score = score + carry*2
                compressed_row.append(carry*2)
                space.append(0)
                carry = None
            else:
                compressed_row.append(carry)
                carry = val

        if carry is not None:
            compressed_row.append(carry)

        space.extend(compressed_row[::-1])
        return [space, score]

    @staticmethod
    def can_compress_row(row):
        for i in range(1, len(row)):
            carry = row[i-1]
            val = row[i]
            if carry == val:
                return True
        return False

    def can_move(self, move):

        # moves cursor across row
        mx, my = move.value['dir']

        # moves down the column
        iy, ix = abs(mx), abs(my)

        # cursor start
        sx, sy = move.value['start']

        x = sx
        y = sy


        while self.in_bounds(sx) and self.in_bounds(sy):
            row = self.get_row(sx, sy, mx, my)
            non_zeros = np.count_nonzero(row)
            if non_zeros == 4:
                if self.can_compress_row(row):
                    return True
            else:
                return True
            [sx, sy] = [sx + ix, sy + iy]

        return False

    def has_lost(self):
        return not (self.can_move(Move.UP) | self.can_move(Move.DOWN) | self.can_move(Move.LEFT) | self.can_move(Move.RIGHT))


def max_index(prediction, n=0):
    ind = np.unravel_index(np.argmax(prediction, axis=None), prediction.shape)
    return ind


game = Game2048()


states = 3
state_size = game.get_state_size()
input_size = state_size * states
model = keras.Sequential()
model.add(keras.layers.Dense(input_size, input_shape=(input_size,), activation=tf.nn.relu, name='Input'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(input_size/2, activation=tf.nn.relu, name='Hidden1'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(input_size/16, activation=tf.nn.relu, name='Hidden2'))
model.add(keras.layers.Dense(4, activation=tf.nn.relu, name='Output'))
model.summary()
model.build()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, momentum=0.0, nesterov=False, name='SGD'),
              loss="mse", metrics=["mae", "acc"])

game.new_game()
print(game.get_state_size())
game.print_board()

top_score = 0
diff = 2
max_diff = 11
diff_counter_init = 5
diff_counter = diff_counter_init
num_games = diff_counter*32
game = Game2048()
n = 0
played = 0
while n < num_games:

    # set difficulty
    if diff_counter <= 0:
        print("NEXT LEVEL OF DIFFICULTY")
        diff_counter = diff_counter_init
        diff = min(max_diff, diff + 1)
        n = 0

    game.end = int(math.pow(2, max_diff))
    game_end = False
    game.new_game()
    total_score = 0
    step = 0
    input = np.ones(shape=(1, input_size))
    while not game_end:
        ## GENERATE NEXT BOARD STATE
        next_state = game.get_board_state()
        input = np.roll(input, state_size, axis=1)
        sel = np.arange(state_size)
        np.put(input, sel, next_state)
        prediction = model.predict(input)

        ## SELECT A MOVE BASED ON THE PREDICTION
        move_sel = prediction[:]
        can_move = np.array([game.can_move(Move.UP), game.can_move(Move.DOWN),
                             game.can_move(Move.LEFT), game.can_move(Move.RIGHT)]).reshape((1, 4))
        possible_moves = np.array([Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]).reshape((1, 4))
        possible_moves = possible_moves[can_move]
        possible_move_select = prediction[can_move]
        idx = max_index(possible_move_select)
        move = possible_moves[idx]
        expected = move.value["expected"]
        mask = move.value["mask"]

        ## TAKE A STEP WITH THE SELECTED MOVE
        score, status = game.step(move)
        total_score = total_score + score
        step = step + 1

        if status == Status.LOST:
            game_end = True
            expected = np.multiply(prediction, mask)
            model.fit(input, expected, verbose=0)

        if game_end:
            print("Game - " + str(played) + " - " + str(total_score))
            game.print_board()
            if total_score > top_score:
                top_score = total_score
                print("Top Score: " + str(top_score))

        expected = expected * (math.pow(score, 2.0) / math.pow(2048.0, 2.0))
        model.fit(input, move_sel, verbose=0)

    print("Game " + str(n) + " complete, " + " score achieved: " + str(total_score))
    game.print_board()
    n = n + 1
    played = played + 1

