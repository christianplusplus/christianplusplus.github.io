"""Games, or Adversarial Search. 
V1.0
"""

import random
from utils import Dict,Struct,infinity,argmax,num_or_str,if_,update
#______________________________________________________________________________
# Minimax Search

def minimax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. """

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a)))
        return v

    # Body of minimax_decision:
    return argmax(game.actions(state),
                  lambda a: min_value(game.result(state, a)))

#______________________________________________________________________________

def alphabeta_full_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    this version searches all the way to the leaves."""

    player = game.to_move(state)

    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -infinity
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = infinity
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alphabeta_search:
    return argmax(game.actions(state),
                  lambda a: min_value(game.result(state, a),
                                      -infinity, infinity))

def max_value(game, state, alpha, beta, depth, cutoff_test, eval_fn, player):
    if cutoff_test(state, depth):
        return eval_fn(state, player)
    v = -infinity
    for a in game.actions(state):
        v = max(v, min_value(game, game.result(state, a),
                             alpha, beta, depth+1, cutoff_test, eval_fn, player))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def min_value(game, state, alpha, beta, depth, cutoff_test, eval_fn, player):
    if cutoff_test(state, depth):
        return eval_fn(state, player)
    v = infinity
    for a in game.actions(state):
        v = min(v, max_value(game, game.result(state, a),
                             alpha, beta, depth+1, cutoff_test, eval_fn, player))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v
        
def alphabeta_search(state, game, d=4, cutoff_test=None, eval_fn=None):
    """Search game to determine best action; use alpha-beta pruning.
    This version cuts off search and uses an evaluation function."""

    player = game.to_move(state)
    
    # Body of alphabeta_search starts here:
    # The default test cuts off at depth d or at a terminal state
    cutoff_test = cutoff_test or (lambda state,depth: depth>d or game.terminal_test(state))
    eval_fn = eval_fn or (lambda state: game.utility(state, player))
    
    return argmax(game.actions(state),
                  lambda a: min_value(game, game.result(state, a),
                                      -infinity, infinity, 0, cutoff_test, eval_fn, player))

#______________________________________________________________________________
# Generic players for Games
    
def query_player(game, state):
    "Make a move by querying standard input."
    #Non-robust, but generic and will work for any game
    game.display(state)
    return num_or_str(input('Your move? '))
    
def random_player(game, state):
    "A player that chooses a legal move at random."
    return random.choice(game.actions(state))

def alphabeta_player(game, state):
    return alphabeta_search(state, game)

def alphabeta_full_player(game, state):
    return alphabeta_full_search(state, game)
    
'''
#A player that uses alphabeta_search intelligently
def intelligent_player(game, state, cut, eval):
    #eval = lambda state: game.compute_in_row(state.board, state.currentMove, state.player)
    #cut = lambda state,depth: depth > 6 or game.terminal_test(state)
    move = alphabeta_search(state, game, d=4, cutoff_test=cut, eval_fn=eval)
    return move
'''
    
def play_game(game, *players):
    """Play an n-person, move-alternating game.
    """
    state = game.initial
    while True:
        for player in players:
            move = player(game, state)
            state = game.result(state, move)
            if game.terminal_test(state):
                game.display(state)
                return game.utility(state, game.to_move(game.initial))

#______________________________________________________________________________
# Game Class

class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display or you 
    can inherit its default method. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        "Return a list of the allowable moves at this point."
        abstract

    def result(self, state, move):
        "Return the state that results from making a move from a state."
        abstract

    def utility(self, state, player):
        "Return the value of this final state to player."
        abstract

    def terminal_test(self, state):
        "Return True if this is a final state for the game."
        return not self.actions(state)

    def to_move(self, state):
        "Return the player whose move it is in this state."
        return state.to_move

    def display(self, state):
        "Print or otherwise display the state."
        print (state)

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

