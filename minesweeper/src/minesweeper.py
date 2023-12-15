import numpy as np


class Minesweeper:

    def __init__(self, width, height, num_bombs):
        self.state = None
        self.bombs = None
        self.solution = None
        self.width = width
        self.height = height
        self.num_bombs = num_bombs
        self.create_new_game(width, height, num_bombs)

    @staticmethod
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
        kernel[kernel_row_start:kernel_row_end, kernel_col_start:kernel_col_end] = array[row_start:row_end,
                                                                                   col_start:col_end]

        return kernel

    def _create_bombs(self, w, h, n):
        self.bombs = np.full((w, h), False)
        selection = np.unravel_index(np.random.choice(w * h, size=n, replace=False), (w, h))
        self.bombs[selection] = True

    def _create_solution(self):
        self.solution = np.full(self.bombs.shape, 0)
        w, h = self.bombs.shape
        for x, y in np.ndindex((w, h)):
            self.solution[x][y] += np.sum(self.extract_kernel_with_padding(self.bombs, (3, 3), (x - 1, y - 1), False))
        return self.solution

    def create_new_game(self, w, h, num_bombs):
        self.width = w
        self.height = h
        self.num_bombs = num_bombs
        self.state = np.full((self.width, self.height), False)
        self._create_bombs(self.width, self.height, self.num_bombs)
        self._create_solution()

    @staticmethod
    def print_board(board):
        # Write solution.
        print('')

        for x in range(0, board.shape[0]):
            for y in range(0, board.shape[1]):
                print(str(board[x][y]) + ' ', end='')
            print('')

    @staticmethod
    def in_bounds(x, y, board):
        return (0 <= x < board.shape[0]) and (0 <= y < board.shape[1])

    def _flood_fill_expansion(self, x, y, visited=None):

        if visited is None:
            assert self.state[x, y] == False
            visited = []

        if not self.in_bounds(x, y, self.state):
            return

        if self.state[x, y]:
            return

        if self.bombs[x][y]:
            self.state[x][y] = True
            return

        if (x, y) in visited:
            return

        self.state[x][y] = True
        if self.solution[x][y] == 0:
            visited.append((x, y))
            neighbours = [(1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
            for dx, dy in neighbours:
                self._flood_fill_expansion(x + dx, y + dy, visited)

    def select(self, x, y):

        if self.state[x, y]:
            return

        if self.has_won() or self.has_lost():
            return

        assert self.state[x, y] == False
        if self.bombs[x][y] == True:
            self.state[x][y] = True
        else:
            self._flood_fill_expansion(x, y)

    def has_won(self):
        return np.all((self.state[1:-1, 1:-1] == True) == (self.bombs[1:-1, 1:-1] == False))

    def has_lost(self):
        return np.any((self.state[:, :] == True) & (self.bombs[:, :] == True))

    def get_actions(self):
        actions = []
        for x, y in np.ndindex(self.state.shape):
            if self.state[x, y]:
                continue

            neighbours = [(1, 1), (1, 0), (1, -1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (0, 1)]
            for dx, dy in neighbours:
                if self.in_bounds(x + dx, y + dy, self.state) and self.state[x + dx, y + dy]:
                    actions.append((x, y))
                    break
        return actions
