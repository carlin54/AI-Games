from tensorflow import keras
import numpy as np
import os
import random
import datetime
from tensorflow.keras.models import load_model

w = 10
h = 10
p = 1

np.random.seed(0)

def extract_kernel_with_padding(array, kernel_size, position, padding=0):
    """
    Extracts a square kernel from a numpy array. Fills out-of-bounds areas with -1.

    :param array: Input numpy array.
    :param kernel_size: Size of the square kernel.
    :param position: Tuple (row, col) indicating the top-left corner of the kernel.
    :return: The extracted kernel with out-of-bounds areas filled with -1.
    """
    row, col = position
    rows, cols = array.shape

    # Initialize a kernel filled with -1
    kernel = np.full(kernel_size, padding)

    # Calculate the slice boundaries
    row_start = max(row, 0)
    row_end = min(row + kernel_size[0], rows)
    col_start = max(col, 0)
    col_end = min(col + kernel_size[1], cols)

    # Calculate the corresponding slice boundaries in the kernel
    kernel_row_start = row_start - row
    kernel_row_end = kernel_row_start + (row_end - row_start)
    kernel_col_start = col_start - col
    kernel_col_end = kernel_col_start + (col_end - col_start)

    # Copy the valid parts of the array into the kernel
    kernel[kernel_row_start:kernel_row_end, kernel_col_start:kernel_col_end] = array[row_start:row_end, col_start:col_end]

    return kernel

def create_bombs(w, h, n):
    bombs = np.full((h, w), False)
    selection = np.unravel_index(np.random.choice(w*h, size=n, replace=False), (w, h))
    bombs[selection] = True
    return bombs


# Create solution as a m+2 x n+2 array.
def create_solution(bombs):
    solution = np.full(bombs.shape, 0)
    w, h = bombs.shape
    for x, y in np.ndindex((w, h)):
        solution[x][y] += np.sum(extract_kernel_with_padding(bombs, (3,3), (x-1, y-1), False))
    return solution


def create_new_game(w, h):
    state = np.full((w, h), False)
    return state


def print_board(board):
    # Write solution.
    print('')
    for x in range(0, board.shape[0]):
        for y in range(0, board.shape[1]):
            print(str(board[x][y]) + ' ', end='')
        print('')


def in_bounds(x, y, board):
    return (0 <= x < board.shape[0]) and (0 <= y < board.shape[1])


def flood_fill_expansion(x, y, state, solution, bombs, visited=None):

    if visited is None:
        assert state[x, y] == False
        get_actions(state, None)
        visited = []

    if not in_bounds(x, y, state):
        return state

    if state[x, y]:
        return state

    if bombs[x][y]:
        return state

    if (x, y) in visited:
        return state

    state[x][y] = True
    if solution[x][y] == 0:
        visited.append((x, y))
        neighbours = [(1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        for dx, dy in neighbours:
            state = flood_fill_expansion(x + dx, y + dy, state, solution, bombs, visited)

    return state


def select(x, y, state, solution, bombs):
    if state[x, y]:
        get_actions(state, None)
    assert state[x, y] == False
    if bombs[x][y] == True:
        state[x][y] = True
    else:
        state = flood_fill_expansion(x, y, state, solution, bombs)

    return state


def has_won(state, bombs):
    return np.all((state[1:-1, 1:-1] == True) == (bombs[1:-1, 1:-1] == False))


def has_lost(state, bombs):
    return np.any((state[:, :] == True) & (bombs[:, :] == True))

def get_actions(state, solution):
    actions = []
    for x, y in np.ndindex(state.shape):
        if state[x, y]:
            continue

        neighbours = [(1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        for dx, dy in neighbours:
            if in_bounds(x + dx, y + dy, state) and not state[x + dx, y + dy]:
                actions.append((x, y))
                break
    return actions


def actions_to_inputs(actions, state, solution, rx, ry):
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, unknown, oob
    inputs = list()
    samples = np.zeros((len(actions), rx * ry * 11))
    for i, action in enumerate(actions):
        x, y = action
        for dx, dy in np.ndindex((rx, ry)):
            px = x + dx - rx//2
            py = y + dy - ry//2
            data_offset = (dx * rx * 11) + dy * 11

            if not in_bounds(px, py, state):
                samples[i, data_offset + 10] = 1
                continue

            if not state[px, py]:
                samples[i, data_offset + 9] = 1
                continue

            index = solution[px, py]
            samples[i, data_offset + index] = 1

    return samples

rx, ry = 7, 7

if os.path.exists('minesweeper_model.h5'):
    model = load_model('minesweeper_model.h5')
else:
    model = keras.Sequential([
        keras.layers.Input(shape=(11 * rx * ry,)),
        keras.layers.Dense(11 * 5 * 5, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


num_games = 10000
won_counter = 0
svc = list()
for i in range(num_games):
    exit = False
    w, h, p = 10, 10, 20
    bombs = create_bombs(w, h, p)
    state = create_new_game(w, h)
    solution = create_solution(bombs)

    x, y = random.choice(np.argwhere(solution == 0))
    #x, y = random.choice(np.argwhere(bombs == False))
    state = select(x, y, state, solution, bombs)
    survival_counter = 1
    while not exit:
        actions = get_actions(state, solution)
        if len(actions) == 0:
            state = select(x, y, state, solution, bombs)
            break
        inputs = actions_to_inputs(actions, state, solution, rx, ry)
        predictions = model.predict(inputs, verbose=None)
        X, Y = zip(*actions)
        outputs = 1.0 - bombs[X, Y].astype(np.float64)
        loss = model.train_on_batch(inputs, outputs)
        sx, sy = actions[np.argmax(predictions)]
        before = np.sum(state)
        state = select(sx, sy, state, solution, bombs)
        after = np.sum(state)
        if after <= before:
            state = select(sx, sy, state, solution, bombs)
            break
        if has_lost(state, bombs):
            exit = True
            y_true = 0
        else:
            y_true = 1
            survival_counter = survival_counter + 1

        if has_won(state, bombs):
            exit = True
            won_counter += 1
    svc.append(survival_counter)
    if len(svc) < num_games // 100:
        print(f"Game {i}, Wins: {float(won_counter)/float(i+1):.2f}, Survival Counter {np.mean(svc):.2f}.")
    else:
        print(f"Game {i}, Wins: {float(won_counter) / float(i + 1):.2f}, Survival Counter {np.mean(svc):.2f}, Rolling Counter {np.mean(svc[num_games//100:]):.2f}.")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f'{timestamp}_minesweeper_model.h5')
