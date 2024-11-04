import random
import numpy as np
from enum import Enum
from collections import deque

BOARD_SIZE = 5

class ECellType(Enum):
    Empty = '#'
    First = 'X'
    Second = 'O'

def RandomCellType():
    return random.choice([ECellType.Empty, ECellType.First, ECellType.Second])

def RandomNonEmptyCellType():
    return random.choice([ECellType.First, ECellType.Second])

class ERowType(Enum):
    Row = 0
    Column = 1

def getRandomRowType():
    return random.choice(list(ERowType))

class EMoveDirection(Enum):
    Down = -1
    Up = 1

def getRandomMoveDirection():
    return random.choice(list(EMoveDirection))

class Move:
    def __init__(self, cellType, rowType, row, column, direction):
        self.cellType = cellType
        self.rowType = rowType
        self.row = row
        self.column = column
        self.direction = direction

def GetNextMove(cellType):
    return ECellType.Second if cellType == ECellType.First else ECellType.First

class GameState:
    BoardSize = BOARD_SIZE

    def __init__(self):
        self.Board_ = np.full((self.BoardSize, self.BoardSize), ECellType.Empty)

    def GetCell(self, i, j):
        return self.Board_[i, j]

    def SetCell(self, i, j, value):
        self.Board_[i,j] = value
        
    def Fill(self, f):
        for i in range(self.BoardSize):
            for j in range(self.BoardSize):
                self.Board_[i, j] = f()

    def SerializeAsString(self):
        gameState = ""
        for i in range(self.BoardSize):
            for j in range(self.BoardSize):
                gameState += str(self.Board_[i, j].value)
            gameState += "\n"
        return gameState

    def ApplyMove(self, move):
        row_number = move.row if move.rowType == ERowType.Row else move.column
        start_point = move.column if move.rowType == ERowType.Row else move.row
        end_point = 0 if move.direction == EMoveDirection.Up else self.BoardSize - 1
        self.SetCell(move.row, move.column, move.cellType)
        self.PermuteRow(row_number, start_point, end_point, move.direction.value, move.rowType)

    def checkFullRow(self, playerSymbol):
        for i in range(self.BoardSize):
            if np.all(self.Board_[i] == playerSymbol) or np.all(self.Board_.T[i] == playerSymbol):
                return True

        left_diagonal = [self.Board_[i][i] for i in range(self.BoardSize)]
        right_diagonal = [self.Board_[i][self.BoardSize-1-i] for i in range(self.BoardSize)]
        if np.all(left_diagonal == playerSymbol) or np.all(right_diagonal == playerSymbol):
            return True
        return False

    def IsFinish(self):
        return self.checkFullRow(ECellType.First) or self.checkFullRow(ECellType.Second)

        
    def PermuteRow(self, row_number, start, end, dir, rowType):
        if start > end:
            start, end = end, start  
        if rowType == ERowType.Row:
            self.Board_[row_number, start:end+1] = np.roll(self.Board_[row_number, start:end+1], dir)
        else: 
            self.Board_[start:end+1, row_number] = np.roll(self.Board_[start:end+1, row_number], dir)


def getRandomStartPosition():
    state = GameState()
    state.Fill(RandomCellType)
    return state


def SerializeMoveAsString(move):
    if move.rowType == ERowType.Row:
        dir = "right" if move.direction == EMoveDirection.Up else "left"
        sliding = f"slide to the {dir}, "
    else:
        dir = "down" if move.direction == EMoveDirection.Up else "up"
        sliding = f"slide {dir}, "
    taking = f"Take the cube from the row {move.row}, column {move.column}, "
    putting = f"putting {move.cellType.value}"
    return f"{taking}{sliding}{putting}\n"



move_to_dir = {
    (ERowType.Row, EMoveDirection.Up): "right",
    (ERowType.Row, EMoveDirection.Down): "left",
    (ERowType.Column, EMoveDirection.Up): "down",
    (ERowType.Column, EMoveDirection.Down): "up",
}

piece_names = ["piece", "cube", "block"]
digit_to_word = ["zero", "one", "two", "three", "four"]
cardinal_to_ordinal = ["zeroth", "first", "second", "third", "fourth"]
cell_names = {ECellType.First: ['X', 'ex', 'cross'], 
              ECellType.Second: ['O', 'oh', 'nought']
              }
dir_names = {"up": ["up", "upward", "north"],
             "down": ["down", "downward", "south"],
             "right": ["right", "rightward", "east"],
             "left": ["left", "leftward", "west"]
             }
row_variants = [lambda y: f"row {y}", 
                lambda y: f"row {digit_to_word[y]}", 
                lambda y: f"{cardinal_to_ordinal[y]} row"
                ]
col_variants = [lambda y: f"column {y}", 
                lambda y: f"column {digit_to_word[y]}", 
                lambda y: f"{cardinal_to_ordinal[y]} column"
                ]
put_variants = ["putting", "placing", "inserting"]
take_variants = ["take", "withdraw", "grab", "pick"]


def SerializeMoveAsString_1(move):
    dir = move_to_dir[(move.rowType, move.direction)]
    slide = f"slide {random.choice(dir_names[dir])}, "
    piece = random.choice(piece_names)
    
    row_col = [random.choice(row_variants)(move.row), random.choice(col_variants)(move.column)]
    first, second = random.choice([(0, 1), (1, 0)])
    
    take = f"{random.choice(take_variants)} the {piece} from the {row_col[first]}, {row_col[second]}, "
    put = f"{random.choice(put_variants)} {random.choice(cell_names[move.cellType])}"
    return f"{take.capitalize()}{slide}{put}.\n"


def GetRandomCell(state, cellType):
    availableCells = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if (i == 0 or j == 0 or i == BOARD_SIZE - 1 or j == BOARD_SIZE - 1) and (state.GetCell(i, j) == cellType or state.GetCell(i, j) == ECellType.Empty):
                availableCells.append((i, j))
    return random.choice(availableCells)


def getRandomMove(state, cellType):
    i, j = GetRandomCell(state, cellType)
    rowType = getRandomRowType()
    
    if i == 0 or i == BOARD_SIZE - 1:
        dir = EMoveDirection.Down if i == 0 else EMoveDirection.Up if rowType == ERowType.Column else getRandomMoveDirection()
    else:
        dir = EMoveDirection.Down if j == 0 else EMoveDirection.Up if rowType == ERowType.Row else getRandomMoveDirection()

    return Move(cellType, rowType, i, j, dir)


STATES_NUMBER = 100

if __name__ == "__main__":
    Q = deque([(GameState(), ECellType.First)])
    for _ in range(STATES_NUMBER):
        state, cellType = Q.popleft()
        if state.IsFinish():
            Q.appendleft((GameState(), ECellType.First))
            print("\n\nOHOHOHOHOHOHO!!!!!\n\n")
        else:
            print("------------\n")
            move = getRandomMove(state, cellType)
            print(state.SerializeAsString() + "\n" + SerializeMoveAsString_1(move))
            state.ApplyMove(move)
            print(state.SerializeAsString())
            Q.append((state, GetNextMove(cellType)))