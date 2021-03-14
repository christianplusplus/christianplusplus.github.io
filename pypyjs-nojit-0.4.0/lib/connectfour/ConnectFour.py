#CS300
#Assignment 5
#Christian Wendlandt
#Professor George Thomas
#12-11-18

import random
from games import Game
from games import alphabeta_full_player,alphabeta_player,alphabeta_search
from utils import Dict,Struct,update,if_,num_or_str,infinity,argmax

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

