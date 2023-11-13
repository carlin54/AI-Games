from tensorflow import keras
import numpy as np

w = 10
h = 10
p = 1

np.random.seed(0)


def create_bombs(w, h, n):
    bombs = np.full((h + 2, w + 2), False)
    while np.sum(bombs) < n:
        x, y = int(np.random.uniform(1, w-1, 1)), int(np.random.uniform(1, h-1, 1))
        if (x == h // 2) and (y == w // 2): continue
        bombs[x][y] = True
    return bombs


# Create solution as a m+2 x n+2 array.
def create_solution(bombs):
    solution = np.full(bombs.shape, 0)
    w, h = bombs.shape
    for x in range(1, w):
        for y in range(1, h):
            if in_bounds(x, y, bombs) and bombs[x][y]:
                solution[x][y] += 1
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
    return (0 <= x < board.shape[0] - 1) and (0 <= y < board.shape[1])


def flood_fill_expansion(x, y, state, solution, bombs, visited=[]):
    if (x, y) in visited:
        return state

    if not in_bounds(x, y, state):
        return state

    if bombs[x][y]:
        return state

    state[x][y] = True
    if solution[x][y] == 0:
        visited.append((x, y))
        neighbours = [(1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        for dx, dy in neighbours:
            state = flood_fill_expansion(x + dx, y + dy, state, solution, bombs, visited)

    return state


def select(x, y, state, solution, bombs):
    if bombs[x][y] == True:
        state[x][y] = True
    else:
        state = flood_fill_expansion(x, y, state, solution, bombs)

    return state


def has_won(state, bombs):
    return np.all((state[1:-1, 1:-1] == True) == (bombs[1:-1, 1:-1] == False))


def has_lost(state, bombs):
    return np.any((state[:, :] == True) & (bombs[:, :] == True))



rx, ry = 1, 1
model = keras.Sequential([
    keras.layers.Input(shape=(11 * (5) * (5),)),
    keras.layers.Dense(11 * 5 * 5, activation='relu'),
    keras.layers.Dense(11 * 5 * 5, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def get_actions(state, solution):
    actions = []
    for x in range(state.shape[0]):
        for y in range(state.shape[1]):
            neighbours = [(1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
            for dx, dy in neighbours:
                if in_bounds(x + dx, y + dy, state) and state[x + dx, y + dy]:
                    actions.append((x, y))
                    break
    return actions


def actions_to_inputs(actions, state, solution, rx, ry):
    # 0, 1, 2, 3, 4, 5, 6, 7, 8, unknown, oob
    inputs = list()

    for action in actions:
        x, y = action
        data = np.zeros((rx * ry * 11,))
        for dx in range(-rx, rx + 1):
            for dy in range(-ry, ry + 1):
                if not in_bounds(x + dx, y + dy, state):
                    data[10] = 1
                    continue

                if not state[x + dx, y + dy]:
                    data[9] = 1
                    continue

                index = solution[x + dx, y + dy]
                data[index] = 1

        inputs.append(tuple([(x, y), data]))

    return inputs

num_games = 100
won_counter = 0

for i in range(num_games):
    exit = False
    w, h, p = 10, 10, 10
    bombs = create_bombs(w, h, p)
    state = create_new_game(w, h)
    solution = create_solution(bombs)
    state = select(w // 2, h // 2, state, solution, bombs)
    survival_counter = 1
    while not exit:
        actions = get_actions(state, solution)
        inputs = actions_to_inputs(actions, state, solution, 2, 2)
        step_input_data = list()
        step_output_data = list()
        predictions = list()
        for action, data in zip(actions, inputs):
            (x, y), data = action
            predictions.append(model.predict(data)
            step_input_data.append(data)
            step_output_data.append(int(bombs[x, y]))

        i = predictions.index(max(predictions))
        x, y = actions[i]
        state = select(x, y, state, solution, bombs)

        if has_lost(state, bombs):
            exit = True
            y_true = 0
        else:
            y_true = 1
            survival_counter = survival_counter + 1

        model.train_on_batch(step_input_data, step_output_data)

        if has_won(state, bombs):
            exit = True
            won_counter += 1

    print(f"Game {i} survival counter {survival_counter}."))
