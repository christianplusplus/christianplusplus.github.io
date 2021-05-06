"""

Provides some widely useful utilities -Based on AIMA code
We will incrementally build this through the course...
"""
import operator, math, random, copy, sys, bisect, re
from functools import reduce

#______________________________________________________________________________
# Simple Data Structures

infinity = 1.0e400

def Dict(**entries):
    """Create a dict out of the argument=value arguments.
    >>> Dict(a=1, b=2, c=3)
    {'a': 1, 'c': 3, 'b': 2}
    """
    return entries  #converts multiple name value pairs to a dictionary

class DefaultDict(dict):
    """Dictionary with a default value for unknown keys."""
    def __init__(self, default):
        self.default = default

    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, copy.deepcopy(self.default))

    def __copy__(self):
        copy = DefaultDict(self.default)
        copy.update(self)
        return copy
        
def update(x, **entries):
    """Update a dict, or an object with slots, according to `entries` dict.
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    """
    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries) #update method updates dictionary with another's keys
    return x

#Added V5.0
class Struct:
    """Create an instance with argument=value slots.
    This is for making a lightweight object whose class doesn't matter."""
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __cmp__(self, other):
        if isinstance(other, Struct):
            return cmp(self.__dict__, other.__dict__)
        else:
            return cmp(self.__dict__, other)

    def __repr__(self):
        args = ['%s=%s' % (k, repr(v)) for (k, v) in vars(self).items()]
        return 'Struct(%s)' % ', '.join(sorted(args))

#______________________________________________________________________________
# Misc Functions
def euclidean(a, b): 
    '''The Euclidean distance between two (x, y) points.'''
    return math.hypot((a[0] - b[0]), (a[1] - b[1]))

def manhattan(a, b): 
    '''The Manhattan distance between two (x, y) points.'''
    return math.fabs(a[0] - b[0])+ math.fabs(a[1] - b[1])

def print_table(table, header=None, key=None):
    """Print a list of lists as a table, so that columns line up nicely.
    header, if specified, will be printed as the first row."""
    if key:
        print (key)
    if header:
        table = [header] + table
    maxlen = lambda seq: max(map(len, seq))
    sizes = map(maxlen, zip(*[map(str, row) for row in table]))
    alignments =[]
    for row in table:
        for (size,x) in zip(sizes,row):
            alignments.append(size) 
    for row in table:
        i=0
        for s in row:
            print(repr(s).ljust(alignments[i]+3), end="")
            i+=1
        print()
    print()

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if not memoized_fn.cache.has_key(args):
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]
        memoized_fn.cache = {}
    return memoized_fn

#Added V5.0
def num_or_str(x):
    """The argument is a string; convert to a number if possible, or strip it.
    >>> num_or_str('42')
    42
    >>> num_or_str(' 42x ')
    '42x'
    """
    if isnumber(x): return x
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()

def if_(test, result, alternative):
    """Like C++ and Java's (test ? result : alternative), except
    both result and alternative are always evaluated. However, if
    either evaluates to a function, it is applied to the empty arglist,
    so you can delay execution by putting it in a lambda.
    >>> if_(2 + 2 == 4, 'ok', lambda: expensive_computation())
    'ok'
    """
    if test:
        if callable(result): return result()
        return result
    else:
        if callable(alternative): return alternative()
        return alternative

def isnumber(x):
    "Is x a number? We say it is if it has a __int__ method."
    return hasattr(x, '__int__')

#______________________________________________________________________________
# Functions on Sequences (mostly inspired by Common Lisp)
# NOTE: Sequence functions (count_if, find_if, every, some) take function
# argument first (like reduce, filter, and map).

def removeall(item, seq):
    """Return a copy of seq (or string) with all occurences of item removed.
    >>> removeall(3, [1, 2, 3, 3, 2, 1, 3])
    [1, 2, 2, 1]
    >>> removeall(4, [1, 2, 3])
    [1, 2, 3]
    """
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]

def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements.
    >>> unique([1, 2, 3, 2, 1])
    [1, 2, 3]
    """
    return list(set(seq))

def product(numbers):
    """Return the product of the numbers.
    >>> product([1,2,3,4])
    24
    """
    return reduce(operator.mul, numbers, 1)

def count_if(predicate, seq):
    """Count the number of elements of seq for which the predicate is true.
    >>> count_if(callable, [42, None, max, min])
    2
    """
    f = lambda count, x: count + (not not predicate(x))
    return reduce(f, seq, 0)

def find_if(predicate, seq):
    """If there is an element of seq that satisfies predicate; return it.
    >>> find_if(callable, [3, min, max])
    <built-in function min>
    >>> find_if(callable, [1, 2, 3])
    """
    for x in seq:
        if predicate(x): return x
    return None

def every(predicate, seq):
    """True if every element of seq satisfies predicate.
    >>> every(callable, [min, max])
    1
    >>> every(callable, [min, 3])
    0
    """
    for x in seq:
        if not predicate(x): return False
    return True

def some(predicate, seq):
    """If some element x of seq satisfies predicate(x), return predicate(x).
    >>> some(callable, [min, 3])
    1
    >>> some(callable, [2, 3])
    0
    """
    for x in seq:
        px = predicate(x)
        if px: return px
    return False

def isin(elt, seq):
    """Like (elt in seq), but compares with is, not ==.
    >>> e = []; isin(e, [1, e, 3])
    True
    >>> isin(e, [1, [], 3])
    False
    """
    for x in seq:
        if elt is x: return True
    return False

#______________________________________________________________________________
# Functions on sequences of numbers
# NOTE: these take the sequence argument first, like min and max,
# and like standard math notation: \sigma (i = 1..n) fn(i)
# A lot of programing is finding the best value that satisfies some condition;
# so there are three versions of argmin/argmax, depending on what you want to
# do with ties: return the first one, return them all, or pick at random.

def argmin(seq, fn):
    """Return an element with lowest fn(seq[i]) score; tie goes to first one.
    >>> argmin(['one', 'to', 'three'], len)
    'to'
    """
    best = seq[0]; best_score = fn(best)
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score
    return best

def argmin_list(seq, fn):
    """Return a list of elements of seq[i] with the lowest fn(seq[i]) scores.
    >>> argmin_list(['one', 'to', 'three', 'or'], len)
    ['to', 'or']
    """
    best_score, best = fn(seq[0]), []
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = [x], x_score
        elif x_score == best_score:
            best.append(x)
    return best

def argmin_random_tie(seq, fn):
    """Return an element with lowest fn(seq[i]) score; break ties at random.
    Thus, for all s,f: argmin_random_tie(s, f) in argmin_list(s, f)"""
    best_score = fn(seq[0]); n = 0
    for x in seq:
        x_score = fn(x)
        if x_score < best_score:
            best, best_score = x, x_score; n = 1
        elif x_score == best_score:
            n += 1
            if random.randrange(n) == 0:
                best = x
    return best

def argmax(seq, fn):
    """Return an element with highest fn(seq[i]) score; tie goes to first one.
    >>> argmax(['one', 'to', 'three'], len)
    'three'
    """
    return argmin(seq, lambda x: -fn(x))

def argmax_list(seq, fn):
    """Return a list of elements of seq[i] with the highest fn(seq[i]) scores.
    >>> argmax_list(['one', 'three', 'seven'], len)
    ['three', 'seven']
    """
    return argmin_list(seq, lambda x: -fn(x))

def argmax_random_tie(seq, fn):
    "Return an element with highest fn(seq[i]) score; break ties at random."
    return argmin_random_tie(seq, lambda x: -fn(x)) 

#______________________________________________________________________________
#Statistical functions
def probability(p):
    "Return true with probability p."
    return p > random.uniform(0.0, 1.0)

def weighted_sample_with_replacement(seq, weights, n):
    """Pick n samples from seq at random, with replacement, with the
    probability of each element in proportion to its corresponding
    weight."""
    sample = weighted_sampler(seq, weights)
    return [sample() for s in range(n)]

def weighted_sampler(seq, weights):
    "Return a random-sample function that picks from seq weighted by weights."
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]
    
#______________________________________________________________________________
# Queues: Stack, FIFOQueue

class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        abstract

    def extend(self, items):
        for item in items: self.append(item)

def Stack():
    """Return an empty list, suitable as a Last-In-First-Out Queue."""
    return []

class FIFOQueue(Queue):
    """A First-In-First-Out Queue."""
    def __init__(self):
        self.A = []; self.start = 0
    def append(self, item):
        self.A.append(item)
    def __len__(self):
        return len(self.A) - self.start
    def extend(self, items):
        self.A.extend(items)
    def pop(self):
        e = self.A[self.start]
        self.start += 1
        if self.start > 5 and self.start > len(self.A)/2:
            self.A = self.A[self.start:]
            self.start = 0
        return e
    def __contains__(self, item):
        return item in self.A[self.start:]

class PriorityQueue(Queue):
    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""
    def __init__(self, order=min, f=lambda x: x):
        update(self, A=[], order=order, f=f)

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return some(lambda x: x[1] == item, self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)
                return

"""Games, or Adversarial Search. 
V1.0
"""

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

#CS300
#Assignment 5
#Christian Wendlandt
#Professor George Thomas
#12-11-18

#____________________________________________________________
# Connect Four

def play_game(game, *players, d):
    """Play an n-person, move-alternating game.
    """
    state = game.initial
    while True:
        for player in players:
            move = player(game, state, d)
            state = game.result(state, move)
            if game.terminal_test(state):
                game.display(state)
                return game.utility(state, game.to_move(game.initial))

#____________________________________________________________
# 

def connect_four_query_player(game, state, depth=None):
    "Make a move by querying standard input."
    tryAgain = True
    while tryAgain:
        game.display(state)
        try:
            val = int(input('Select a column ranging from 1-7.\n'))
        except ValueError:
            print('Not a number, try again')
            continue
        if (val>=1 and val<=7):
            if val in state.moves:
                tryAgain=False
            else:
                print('Illegal move, try again')
        else:
            print('Input out of bounds, try again')
    return val

#____________________________________________________________
# 

def random_player(game, state, depth=None):
    "A player that chooses a legal move at random."
    return random.choice(game.actions(state))

#____________________________________________________________
#   
 
def intelligent_player(game, state, depth=4):
    return alphabeta_search(state, game, d=depth, eval_fn=intel_eval_fn)
    
def intel_eval_fn(state, player):
    score = 0
    #horz
    for y in range(1, 7):
        list = []
        for x in range(1, 8):
            list.append(state.board.get((x,y)))
        score += lineScore(list, player)
    #vert
    for x in range(1, 8):
        list = []
        for y in range(1, 7):
            list.append(state.board.get((x,y)))
        score += lineScore(list, player)
    #/diag
    for i in range(6):
        x = max(1,i-1)
        y = max(1,3-i)
        list = []
        while(x <= 7 and y <= 6):
            list.append(state.board.get((x,y)))
            x += 1
            y += 1
        score += lineScore(list, player)
    #\diag
    for i in range(6):
        x = min(7,i+4)
        y = max(1,i-2)
        list = []
        while(x >= 1 and y <= 6):
            list.append(state.board.get((x,y)))
            x -= 1
            y += 1
        score += lineScore(list, player)
    return score

def lineScore(line, player):
    score = 0
    for sublist in [[line[index-3],line[index-2],line[index-1],line[index]] for index in range(3, len(line))]:
        countGood = countBad = 0
        for mark in sublist:
            if mark == player:
                countGood += 1
            elif mark == oppPlayer(player):
                countBad += 1
        score += (1 << countGood) - (1 << countBad)
        if countGood == 4:
            return infinity
        if countBad == 4:
            return -infinity
    return score
        

def oppPlayer(player):
    return 'O' if player == 'X' else 'X'
#____________________________________________________________
#

class ConnectFour(Game):

    def __init__(self, h=7, v=6, k=4):
        update(self, h=h, v=v, k=k)
        self.initial = Struct(to_move = 'X', board = {}, moves = list(range(1, h+1)), lastCell = (0,0))

    def actions(self, state):
        "Legal moves are any column not yet filled."
        return state.moves

    def result(self, state, move):
        if move not in state.moves:
            print("Illegal Move")
            return state # Illegal move has no effect
        board = state.board.copy()
        cell = self.findTopOfColumn(board, move)
        board[cell] = state.to_move
        moves = state.moves.copy()
        if cell[1] == self.v:
            moves.remove(move)
        return Struct(to_move = oppPlayer(state.to_move),
                      board = board, 
                      moves = moves,
                      lastCell = cell)
                      
    def findTopOfColumn(self, board, column):
        search = (column, 1)
        while(search in board):
            search = (search[0], search[1] + 1)
        return search

    def utility(self, state, player):
        "Return the value to player; 1 for win, -1 for loss, 0 otherwise."
        if state.lastCell == (0, 0):
            return 0
        for delta in [(0, 1), (1, 0), (1, -1), (1, 1)]:
            util = self.k_in_row(state.board, state.lastCell, player, delta)
            if util != 0:
                return util
        return 0
    
    def terminal_test(self, state):
        "A state is terminal if it is won or there are no unfilled columns."
        return self.utility(state, state.to_move) != 0 or not state.moves
    

    def display(self, state):
        board = state.board
        for y in reversed(range(1, self.v+1)):
            print('+---+---+---+---+---+---+---+')
            for x in range(1, self.h+1):
                print('|',end='')
                print(' ' + board.get((x, y), ' '),end=" ")
            print('|')
        print('+---+---+---+---+---+---+---+')

    def k_in_row(self, board, move, player, delta):
        "Return 1 for player win, -1 for player loss, 0 for neither yet."
        charInRow = board[move];
        x, y = move
        n = 0 # n is number of moves in row
        while board.get((x, y)) == charInRow:
            n += 1
            x, y = x + delta[0], y + delta[1]
        x, y = move[0] - delta[0], move[1] - delta[1]
        while board.get((x, y)) == charInRow:
            n += 1
            x, y = x - delta[0], y - delta[1]
        if n >= self.k:
            return 1 if charInRow == player else -1
        else:
            return 0
                
#____________________________________________________________
#

def tournament(numberOfMatches, depth):
    IvRwins = IvRties = IvRlosses = IvIwins = IvIties = IvIlosses = 0
    for i in range(numberOfMatches):
        winner = play_game(ConnectFour(), intelligent_player, random_player, d=depth)
        if(winner == 1):
            IvRwins += 1
        elif(winner == 0):
            IvRties += 1
        else:
            IvRlosses += 1
        #winner = play_game(ConnectFour(), intelligent_player, intelligent_player, d=depth)
        #winner = play_game(ConnectFour(), intelligent_player, intelligent_player, d=depth)
        #if(winner == 1):
        #    IvIwins += 1
        #elif(winner == 0):
        #    IvIties += 1
        #else:
        #    IvIlosses += 1
        print('{0:.0f}% complete\n'.format((i + 1) / numberOfMatches * 100))
    file = open('results.txt', 'w')
    file.write(str(numberOfMatches) + ' games played\n')
    file.write('Intelligent player wins against Random player: {0:.0f}%\n'.format(IvRwins / numberOfMatches * 100))
    file.write('Intelligent player ties against Random player: {0:.0f}%\n'.format(IvRties / numberOfMatches * 100))
    file.write('Intelligent player loses against Random player: {0:.0f}%\n'.format(IvRlosses / numberOfMatches * 100))
    #file.write('Intelligent player wins against Intelligent player: {0:.0f}%\n'.format(IvIwins / numberOfMatches * 100))
    #file.write('Intelligent player ties against Intelligent player: {0:.0f}%\n'.format(IvIties / numberOfMatches * 100))
    #file.write('Intelligent player loses against Intelligent player: {0:.0f}%\n'.format(IvIlosses / numberOfMatches * 100))
    

def main():
    depth = 4
    if 1 == play_game(ConnectFour(), connect_four_query_player, intelligent_player, d=depth):
        print("Congradulations! You won!")
    else:
        print("Sorry, you lost.");
    #tournament(10,depth)
    
main()
print("Hit [ENTER] to terminate the program.")
input()

"""
print(play_game(TicTacToe(), tic_tac_query_player,random_player))
print(play_game(TicTacToe(),tic_tac_query_player,alphabeta_player))
print(play_game(TicTacToe(), random_player, random_player)) 
print(play_game(TicTacToe(), alphabeta_player,alphabeta_player)) 
"""