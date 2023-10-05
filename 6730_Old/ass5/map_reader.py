# coding: utf-8
# COMP1730/6730 S1 2023 --- Homework 5
# Submission is due 11:55pm, Saturday the 20th of May, 2023.

# YOUR ANU ID: u7095197
# YOUR NAME: Yangbohan Miao

# You should implement two function read_map and get_spot; you may define
# more functions if this will help you to achieve the functional
# correctness, and to improve the code quality of you program;
# you should modify test functions (functions with names which start with "test_")
# but you can add more test functions of your own


# import sys
import numpy as np

## DO NOT DELETE THESE STATEMENTS -- THEY ARE USED IN TEST FUNCTIONS
from io import StringIO
from unittest.mock import patch, mock_open
dummy_map_file = 'dummy_map.txt'
the_module = 'map_reader'


def read_map(map_file):
    '''
    should return a nested sequence with 0 or 1 in the inner sequences;
    alternatively, return a 2D numpy array;
    the dimensions (shape in the ndarray case) are to be determined by the
    map_file contents
    THE ABOVE LINES MUST BE REPLACED BY PROPER DOCSTRING
    '''
    # with open(map_file) as infile:
    #     return [[0]], 0.0
    with open(map_file, 'r') as infile:
        lines = infile.readlines()

    world_map = []
    for line in lines:
        map_line = []
        for c in line:
            if c == ' ':
                map_line.append(0)
            elif c == '\n':
                continue
            else:
                map_line.append(1)
        world_map.append(map_line)
    #print(world_map)
    total_land = 0
    for i in range(len(world_map)):
        for j in range(len(world_map[i])):
            total_land += world_map[i][j]

    #print(total_land)
    total_area = len(world_map) * len(world_map[0])
    #print(total_area)
    return np.array(world_map), total_land / total_area


def get_spot(world_map, x, y):
    '''
    returns ('land', 'water', 'coast' or 'seashore') depending
    on the location of the point (x, y), latitude and longitude
    '''
    #return 'terra-incognita'
    max_lat, max_long = 90, 180
    rows, cols = world_map.shape
    #print(rows, cols)
    #print(x, y)
    i = int((max_lat - x) / (2 * max_lat) * (rows-1))
    j = int((y + max_long) / (2 * max_long) * (cols-1))
    #print(i, j)

    land = 0
    water = 0

    for k in range(-1, 2):
        for l in range(-1, 2):
            if i + k < 0:
                posx = 0
            elif i+k >= rows:
                posx = rows - 1
            else:
                posx = i + k
            if j + l < 0:
                posy = 0
            elif j+l >= cols:
                posy = cols - 1
            else:
                posy = j + l
            if posx == i and posy == j:
                continue

            if world_map[posx][posy] == 0:
                water += 1
            else:
                land += 1

    if world_map[i][j] == 0:
        if land == 0:
            return 'water'
        else:
            return 'seashore'
    if world_map[i][j] == 1:
        if water == 0:
            return 'land'
        else:
            return 'coast'







# test for mini-world 1
mock_input = ' ' * 5 + '\n' + 3*(' ' + 'x'*3 + ' \n') + ' ' * 5 + '\n'
sample_data = StringIO(mock_input)
@patch(f'{the_module}.open', return_value=sample_data)
def test_read_map(mock_open):
    expected = [[0]*5]
    expected.extend([[0,1,1,1,0]]*3)
    expected.extend([[0]*5])
    expected = np.asarray(expected, dtype=int)
    # print(expected)
    actual, frac = read_map(dummy_map_file)
    assert abs(frac - 9/25) < 1e-8
    assert mock_open.called
    assert np.array_equal(actual, expected)


# test for mini-world 2
mock_input_2 = '\n'.join([' '*9,
                          '    o    ',
                          '  |   J  ',
                          ' 7     , ',
                          '  ;   o  ',
                          '    _    ',
                          ' '*9]) #+ '\n'
sample_data_2 = StringIO(mock_input_2)
@patch(f'{the_module}.open', return_value=sample_data_2)
def test_read_map_2(mock_open):
    expected = [[0]*9]
    expected.append([0,0,0,0,1,0,0,0,0])
    expected.append([0,0,1,0,0,0,1,0,0])
    expected.append([0,1,0,0,0,0,0,1,0])
    expected.append([0,0,1,0,0,0,1,0,0])
    expected.append([0,0,0,0,1,0,0,0,0])
    expected.append([0]*9)
    expected = np.asarray(expected, dtype=int)
    # print(expected)
    actual, frac = read_map(dummy_map_file)
    assert abs(frac - 8/63) < 1e-8
    assert mock_open.called
    assert np.array_equal(actual, expected)


def test_get_spot():
    # test for mini-world 3
    world = [[0]*5]
    world.extend([[0,1,1,1,0]]*3)
    world.extend([[0]*5]*2)
    world = np.asarray(world, dtype=int)
    assert get_spot(world, -90, 0) == 'water'  # south pole
    assert get_spot(world, 0, -90) == 'coast' # equator, pacific
    assert get_spot(world, 0, -120) == 'seashore' # equator, pacific
    assert get_spot(world, 0, 0) == 'land'    # centre of the world
    assert get_spot(world, 45, -90) == 'coast' # top-left corner of the continent
    assert get_spot(world, 90, -180) == 'seashore' # ocean shore near top-left corner

    print('All tests passed')