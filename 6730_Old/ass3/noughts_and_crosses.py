# COMP1730/6730 Homework 3.

# YOUR ANU ID: u7095197
# YOUR NAME: Yangbohan Miao

def ttt_check(game):
    # diagnal is used to determine if the diagonal of the matrix is all cross or nought
    diagonal = sum(game[i][i] for i in range(len(game)))
    # reverse_diagonal is used to determine if the reverse diagonal of the matrix is all cross or nought
    reverse_diagonal = sum(game[i][len(game) - i - 1] for i in range(len(game)))
    if diagonal == -3 or reverse_diagonal == -3:
        return "Crosses"
    if diagonal == 3 or reverse_diagonal == 3:
        return "Noughts"
    for i in range(len(game)):
        if game[0][i] == game[1][i] == game[2][i] == -1 or sum(game[i]) == -3:
            return "Crosses"
        if game[0][i] == game[1][i] == game[2][i] == 1 or sum(game[i]) == 3:
            return "Noughts"
    return "Draw"


def test_ttt_check():
    '''
    for test grids (final game state),
    cross 'x' corresponds to -1
    nought 'o' corresponds to 1
    empty cell '_' corresponds to 0
    '''
    #  x o _
    #  x x o
    #  o _ x
    game_1 = [[-1, 1, 0], [-1, -1, 1], [1, 0, -1]]
    assert ttt_check(game_1) == "Crosses"
    #  x o x
    #  o x x
    #  o x o
    game_2 = [[-1, 1, -1], [1, -1, -1], [1, -1, 1]]
    assert ttt_check(game_2) == "Draw"
    #  x x o
    #  x _ o
    #  _ _ o
    game_3 = [[-1, -1, 1], [-1, 0, 1], [0, 0, 1]]
    assert ttt_check(game_3) == "Noughts"

    print('All tests passed')

# test_ttt_check()
