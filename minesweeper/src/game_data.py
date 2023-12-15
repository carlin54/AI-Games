class GameData:

    def __init__(self, bombs, solution, actions=None, states=None):

        if actions is None:
            actions = []

        if states is None:
            states = []

        self.bombs = bombs
        self.solution = solution
        self.actions = actions
        self.states = states

    def add_step(self, action_set, state):
        self.actions.append(action_set)
        self.states.append(state)