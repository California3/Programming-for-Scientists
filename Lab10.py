def unnest(alist):
    result = []
    for ele in alist:
        # if type(ele) == list:
        #     result += unnest(ele)
        # else:
        #     result.append(ele)
        result += unnest(ele) if type(ele) == list else [ele]
    return result
 
 
def test_unnest():
    """
    This function runs a number of tests of the unnest function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    """
    
    assert unnest([2, 1, 3, [0, 4]]) == [2, 1, 3, 0, 4]
    assert unnest([1, [3], [2, 4], 0]) == [1, 3, 2, 4, 0]
    assert unnest([[[3, 0], 1], 4, 2]) == [3, 0, 1, 4, 2]
    assert unnest([1, [2], [[3], [[4], 5]]]) == [1, 2, 3, 4, 5]
    assert unnest([0, [[2, [1], 4]], [[3]]]) == [0, 2, 1, 4, 3]
    assert unnest([[[0], 2], 3, 1, 4]) == [0, 2, 3, 1, 4]
    assert unnest([[9, 5, 0, 4], [8, 7, 1], 6, 3, 2]) == [9, 5, 0, 4, 8, 7, 1, 6, 3, 2]
    assert unnest([6, 9, [2, 8, 7, 4], [[0, [5]], 1, 3]]) == [6, 9, 2, 8, 7, 4, 0, 5, 1, 3]
 
    assert unnest([[0], [[[2, 4, 3]], [1]]]) == [0, 2, 4, 3, 1]
    assert unnest([[4, [[1]]], 0, 2, 3]) == [4, 1, 0, 2, 3]
    assert unnest([[[1, 3, 4, [[[[2]]]]]], 0]) == [1, 3, 4, 2, 0]
    assert unnest([[4], 1, [[3, [0], [[2]]]]]) == [4, 1, 3, 0, 2]
    assert unnest([[[0]], 4, [[[3]]], [1, 2]]) == [0, 4, 3, 1, 2]
    assert unnest([7, [[5], [2], 4], 6, [[[0, [8], 1]], 9], [[3]]]) == [7, 5, 2, 4, 6, 0, 8, 1, 9, 3]
    assert unnest([[2, 6, [[[5]]], [7], 4, 9, 1, 0, 8], [[3]]]) == [2, 6, 5, 7, 4, 9, 1, 0, 8, 3]
    assert unnest([8, 6, 2, 1, 5, 7, 3, 9, [[[[[[[4]]]]]]], [0]]) == [8, 6, 2, 1, 5, 7, 3, 9, 4, 0]
    assert unnest([[4, [[[1]], 5, 2, 8, [[[3]], 0, 6]], 7, 9]]) == [4, 1, 5, 2, 8, 3, 0, 6, 7, 9]
    assert unnest([[[[1, 9], [3]], [2, [7, 5, 8], 6, 0]], 4]) == [1, 9, 3, 2, 7, 5, 8, 6, 0, 4]
 
    assert unnest([1, [], [2], [[3], [], [[4], [], 5]]]) == [1, 2, 3, 4, 5]
    assert unnest([1, [[3], []], [], [[], 2, 4], 0]) == [1, 3, 2, 4, 0]
    assert unnest([0, [[], [2, [1], 4]], [[], [3]]]) == [0, 2, 1, 4, 3]
    assert unnest([[], [[], [[], 3, 0], 1], [], 4, 2]) == [3, 0, 1, 4, 2]
    assert unnest([[[0], [], 2], [], [], 3, 1, [], 4]) == [0, 2, 3, 1, 4]
    assert unnest([2, [[]], 1, [3], [[0, 4]]]) == [2, 1, 3, 0, 4]
    assert unnest([[[]]]) == []
 
    print("all tests passed")

test_unnest()

 
# Implement the function count_dict_difference below.
# You can define other functions if it helps you decompose and solve
# the problem.
# Do not import any module that you do not use!
 
# Remember that if this were an exam problem, in order to be marked
# this file must meet certain requirements:
# - it must contain ONLY syntactically valid python code (any syntax
#   or indentation error that stops the file from running would result
#   in a mark of zero);
# - you MAY NOT use global variables; the function must use only the
#   input provided to it in its arguments.
 
def count_dict_difference(A, B):
    results = {}
    keys_of_B = B.keys()
    for key, value in A.items():
        if key not in keys_of_B:
            results[key] = value
        else:
            diff = value - B[key]
            if diff > 0:
                results[key] = value - B[key]
    return results
 
 
def test_count_dict_difference():
    """
    This function runs a number of tests of the count_dict_difference function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    """
 
    assert count_dict_difference({'d': 3, 'e': 1, 'z': 1, 's': 1, 'i': 1, 'r': 1, 'a': 2, 'n': 1, 't': 1}, {'e': 2, 'x': 1, 'g': 1, 's': 1, 'p': 1, 'i': 1, 't': 1, 'a': 2, 'n': 1, 'r': 1}) == {'z': 1, 'd': 3}
    assert count_dict_difference({'m': 1, 'o': 1, 'c': 2, 'r': 1, 'i': 2, 't': 1, 'a': 2, 'n': 1, 'l': 2, 'u': 1}, {'m': 2, 'o': 2, 'c': 1, 'z': 1, 'a': 2, 'i': 5, 'u': 1, 'r': 2, 'n': 2, 's': 1, 't': 2}) == {'l': 2, 'c': 1}
    assert count_dict_difference({'g': 1, 'c': 1, 'a': 2, 'i': 2, 'u': 1, 'r': 1, 'n': 1, 'l': 1, 't': 2}, {'g': 1, 'c': 1, 's': 1, 'a': 2, 'i': 2, 't': 2, 'r': 2, 'l': 2, 'u': 2}) == {'n': 1}
    assert count_dict_difference({'o': 1, 's': 5, 'i': 2, 'a': 3, 'n': 2, 't': 1}, {'o': 1, 'c': 1, 'z': 1, 'i': 2, 't': 2, 'a': 3, 'n': 1, 'l': 1, 'u': 1}) == {'s': 5, 'n': 1}
    assert count_dict_difference({'o': 2, 'c': 1, 'e': 2, 's': 2, 'r': 2, 'n': 2, 't': 1}, {'d': 2, 'c': 1, 'e': 3, 'a': 1, 't': 1, 'r': 2, 'o': 2, 'v': 1}) == {'s': 2, 'n': 2}
    assert count_dict_difference({'e': 4, 'g': 2, 's': 5, 'a': 1, 'i': 1, 'r': 1, 'n': 1, 'v': 1}, {'o': 1, 'i': 1, 'g': 1, 'c': 2, 'e': 1, 'a': 2, 'k': 1, 'u': 1, 'r': 1, 'n': 2, 't': 3}) == {'s': 5, 'e': 3, 'g': 1, 'v': 1}
    
    assert count_dict_difference({0: 1, 17: 2, 2: 1, 19: 2, 4: 1, 8: 2, 18: 1, 13: 1, 14: 1, 15: 1}, {0: 2, 17: 1, 2: 1, 19: 2, 4: 1, 8: 2, 20: 1, 11: 1, 13: 1, 14: 1, 15: 1}) == {17: 1, 18: 1}
    assert count_dict_difference({0: 1, 17: 1, 18: 2, 19: 1, 4: 1, 8: 4, 11: 1, 12: 1}, {0: 1, 17: 2, 2: 1, 19: 1, 4: 2, 8: 1, 20: 1, 11: 2, 12: 1, 13: 1, 14: 1}) == {8: 3, 18: 2}
    assert count_dict_difference({0: 1, 17: 1, 18: 3, 3: 1, 4: 3, 6: 2, 13: 1}, {0: 2, 17: 1, 2: 1, 3: 1, 4: 1, 8: 1, 24: 1, 18: 1, 12: 1, 13: 1, 14: 1}) == {18: 2, 4: 2, 6: 2}
    assert count_dict_difference({0: 3, 3: 1, 4: 1, 6: 1, 11: 1, 13: 1, 14: 1, 18: 1, 19: 1, 20: 1, 21: 1, 24: 1}, {17: 1, 18: 1, 3: 1, 4: 2, 21: 1, 6: 1, 8: 1, 20: 1, 11: 1, 13: 1, 14: 1}) == {0: 3, 24: 1, 19: 1}
    assert count_dict_difference({17: 2, 2: 1, 3: 2, 4: 2, 13: 1, 18: 1, 14: 2, 15: 1}, {0: 1, 17: 2, 2: 1, 4: 2, 5: 1, 18: 1, 12: 1, 13: 3, 14: 2, 15: 1}) == {3: 2}
    assert count_dict_difference({0: 1, 18: 6, 4: 2, 11: 1, 12: 1, 13: 1}, {0: 1, 17: 1, 2: 1, 19: 1, 4: 2, 6: 1, 11: 1, 12: 1, 13: 1, 14: 2}) == {18: 6}
 
    assert count_dict_difference({'in': 1, 'ti': 1, 'iv': 1, 'se': 1, 've': 1, 'en': 1, 'ns': 2, 'it': 1, 'si': 1}, {'ve': 1, 'ti': 1, 'iv': 1, 'si': 1, 'it': 1, 'ns': 2, 'st': 1, 'ra': 1, 'tr': 1, 'in': 1, 'an': 1}) == {'se': 1, 'en': 1}
    assert count_dict_difference({'th': 1, 'gt': 1, 'le': 1, 'en': 2, 'ng': 1, 'he': 1, 'ed': 1, 'ne': 1}, {'th': 1, 'ed': 1, 'ng': 1, 'en': 2, 'gt': 1, 'st': 1, 'he': 1, 're': 1, 'tr': 1, 'ne': 1}) == {'le': 1}
    assert count_dict_difference({'sm': 2, 'ri': 1, 'me': 2, 'es': 1, 'is': 1, 'er': 1}, {'di': 1, 'er': 1, 'st': 1, 'is': 1, 're': 1, 'dn': 1, 'si': 1, 'ne': 1, 'in': 1, 'ed': 1, 'nt': 1, 'es': 3, 'se': 1, 'ss': 1, 'te': 2}) == {'me': 2, 'ri': 1, 'sm': 2}
    assert count_dict_difference({'iz': 1, 'on': 1, 'al': 1, 'ze': 1, 'li': 1, 'at': 1, 'na': 2, 'ti': 1, 'io': 1}, {'za': 1, 'ra': 1, 'on': 2, 'al': 1, 'li': 1, 'iz': 1, 'at': 2, 'na': 1, 'ti': 2, 'io': 2}) == {'ze': 1, 'na': 1}
 
    assert count_dict_difference({(0, 5, 6): 1, (0, 5): 1, (5, 5): 2, (5, 0): 1, (5, 6): 1, (5, 5, 0): 1, (6, 5, 5): 1, (5, 5, 5): 1, (6, 5): 1, (5, 6, 5): 1}, {(0, 5, 5): 1, (5, 5, 6): 1, (0, 5): 1, (5, 5): 2, (5, 0): 1, (5, 6): 1, (6, 5, 0): 1, (5, 5, 5): 1, (6, 5): 1, (5, 6, 5): 1}) == {(0, 5, 6): 1, (5, 5, 0): 1, (6, 5, 5): 1}
    assert count_dict_difference({(2, 0): 1, (7, 7, 7): 2, (8, 7): 1, (8, 7, 7): 1, (7, 2, 0): 1, (7, 7): 3, (7, 2): 1, (7, 7, 2): 1}, {(8, 7, 7): 1, (2, 0): 1, (7, 2, 0): 1, (8, 7): 1, (7, 8, 7): 1, (7, 7, 8): 1, (7, 8): 1, (7, 7): 2, (7, 2): 1, (7, 7, 2): 1}) == {(7, 7, 7): 2, (7, 7): 1}
    assert count_dict_difference({(8, 8, 5): 1, (0, 8, 8): 1, (8, 8): 1, (8, 5, 3): 1, (5, 3, 0): 1, (3, 0): 1, (3, 0, 8): 1, (0, 8): 2, (8, 5): 1, (5, 3): 1}, {(8, 8, 5): 1, (3, 0, 0): 1, (8, 8): 2, (8, 5, 3): 1, (5, 3, 0): 1, (3, 0): 1, (8, 8, 8): 1, (8, 5): 1, (0, 0): 1, (5, 3): 1}) == {(3, 0, 8): 1, (0, 8, 8): 1, (0, 8): 2}
    
    print("all tests passed")

test_count_dict_difference() 

 
# Implement the function approximate_integral below.
# (The statement "pass" is just a placeholder that does nothing: you
# should replace it.)
# You can define other functions if it helps you decompose and solve
# the problem.
# Do not import any module that you do not use!
 
# Remember that if this were an exam problem, in order to be marked
# this file must meet certain requirements:
# - it must contain ONLY syntactically valid python code (any syntax
#   or indentation error that stops the file from running would result
#   in a mark of zero);
# - you MAY NOT use global variables; the function must use only the
#   input provided to it in its arguments.
 
def approximate_integral(lower, upper, nterms):
    def f(x):
        return x**3
    
    area_sum = 0
    d = (upper - lower) / nterms
    for i in range(nterms):
        x = lower + i * d
        x_d = lower + (i+1) * d

        fx = f(x)
        fx_d = f(x_d)

        area_sum += (fx + fx_d) * d / 2
    
    return area_sum
 
 
def test_approximate_integral():
    """
    This function runs a number of tests of the approximate_integral function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    """
    
    assert abs(approximate_integral(0, 1, 1) - 0.5) < 1e-6, 'sum of 0.5'
    assert abs(approximate_integral(1, 2, 1) - 4.5) < 1e-6, 'sum of 4.5'
    assert abs(approximate_integral(0, 2, 1) - 8.0) < 1e-6, 'sum of 8.0'
    assert abs(approximate_integral(0, 1, 2) - 0.3125) < 1e-6, 'sum of 0.03125, 0.28125'
    assert abs(approximate_integral(1, 2, 2) - 3.9375) < 1e-6, 'sum of 1.09375, 2.84375'
    assert abs(approximate_integral(0, 2, 2) - 5.0) < 1e-6, 'sum of 0.5, 4.5'
    assert abs(approximate_integral(0, 1, 5) - 0.26) < 1e-6, 'sum of 0.0008000000000000003, 0.007200000000000002, 0.028000000000000014, 0.07280000000000002, 0.1512'
    assert abs(approximate_integral(1, 2, 5) - 3.7799999999999994) < 1e-6, 'sum of 0.2728, 0.4472, 0.6839999999999998, 0.9927999999999998, 1.3831999999999995'
    assert abs(approximate_integral(0, 2, 5) - 4.16) < 1e-6, 'sum of 0.012800000000000004, 0.11520000000000004, 0.44800000000000023, 1.1648000000000003, 2.4192'
 
    assert abs(approximate_integral(-1, 0, 1) - -0.5) < 1e-6, 'sum of -0.5'
    assert abs(approximate_integral(-2, -1, 1) - -4.5) < 1e-6, 'sum of -4.5'
    assert abs(approximate_integral(-2, 0, 1) - -8.0) < 1e-6, 'sum of -8.0'
    assert abs(approximate_integral(-1, 0, 2) - -0.3125) < 1e-6, 'sum of -0.28125, -0.03125'
    assert abs(approximate_integral(-2, -1, 2) - -3.9375) < 1e-6, 'sum of -2.84375, -1.09375'
    assert abs(approximate_integral(-2, 0, 2) - -5.0) < 1e-6, 'sum of -4.5, -0.5'
    assert abs(approximate_integral(-1, 0, 5) - -0.260) < 1e-6, 'sum of -0.1512, -0.07280, -0.0280, -0.00720, -0.00080'
    assert abs(approximate_integral(-2, -1, 5) - -3.780) < 1e-6, 'sum of -1.38320, -0.99280, -0.6840, -0.44720, -0.27280'
    assert abs(approximate_integral(-2, 0, 5) - -4.160) < 1e-6, 'sum of -2.4192, -1.16480, -0.4480, -0.11520, -0.01280'
 
    assert abs(approximate_integral(-1, 1, 1) - 0.0) < 1e-6, 'sum of 0.0'
    assert abs(approximate_integral(-1, 1, 2) - 0.0) < 1e-6, 'sum of -0.5, 0.5'
    assert abs(approximate_integral(-1, 1, 4) - 0.0) < 1e-6, 'sum of -0.28125, -0.03125, 0.03125, 0.28125'
    assert abs(approximate_integral(-2, 2, 1) - 0.0) < 1e-6, 'sum of 0.0'
    assert abs(approximate_integral(-2, 2, 2) - 0.0) < 1e-6, 'sum of -8.0, 8.0'
    assert abs(approximate_integral(-2, 2, 4) - 0.0) < 1e-6, 'sum of -4.5, -0.5, 0.5, 4.5'
 
    print("all tests passed")

test_approximate_integral()

def interval_intersection(lA, uA, lB, uB):
    if uA < lB or uB < lA:
        return 0.0
    else:
        return min(uA, uB) - max(lA, lB)
 
 
def test_interval_intersection():
    """
    This function runs a number of tests of the interval_intersection function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    """
 
    assert interval_intersection(0, 2, 4, 7.5) == 0.0, "no intersection (uA < lB)"
    assert interval_intersection(1, 3, 2.5, 6) == 0.5, "intersection is [2.5, 3]"
    assert interval_intersection(1, 3, 1.5, 5) == 1.5, "intersection is [1.5, 3]"
    assert interval_intersection(0, 2, -2, 1.5) == 1.5, "intersection is [0, 1.5]"
    assert interval_intersection(1, 3, 0, 3.5) == 2.0, "A is contained in B"
    assert interval_intersection(1.5, 3.5, 0, 3.5) == 2.0, "A is contained in B"
 
    print("all tests passed")

test_interval_intersection()

def super_increasing(seq):
    for i in range(1, len(seq)):
        if sum(seq[:i]) >= seq[i]:
            return False
    return True
 
 
def test_super_increasing():
    """
    This function runs a number of tests of the super_increasing function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    """
 
    assert not super_increasing((1, 3, 5, 7, 19)), "sum(1, 3, 5) = 9 >= 7"
    assert super_increasing([1, 3, 5, 11, 21]), "sum(1) = 1 < 3; sum(1,3) = 4 < 5; sum(1, 3, 5) = 9 < 11; sum(1, 3, 5, 11) = 20 < 21"
    assert super_increasing((0, 1, 2, 4)), "sum(0) = 0 < 1; sum(0, 1) = 1 < 2; sum(0, 1, 2) = 3 < 4"
    assert not super_increasing([0, 0, 1, 2]), "sum(0) = 0 >= 0"
    assert super_increasing((-1, 0, 0, 1)), "sum(-1) = -1 < 0; sum(-1, 0) = -1 < 0; sum(-1, 0, 0) = -1 < 1"
    assert not super_increasing((1, 2, 0, 4)), "sum(1, 2) = 3 >= 0"
    assert super_increasing((-1, 3, 4)), "sum(-1) < 3; sum(-1, 3) = 2 < 4"
    assert not super_increasing((-1, 3, 4, 5)), "sum(-1, 3, 4) = 6 >= 5"
    assert super_increasing((-2, -1, -2)), "sum(-2) < -1; sum(-2, -1) = -3 < -2"
    assert not super_increasing((-2, -1, -4)), "sum(-2, -1) = -3 >= -4"
 
    print("all tests passed")

test_super_increasing()


# Implement the function moving_average below.
# You can define other functions if it helps you decompose and solve
# the problem.
# Do not import any module that you do not use!

# Remember that if this were an exam problem, in order to be marked
# this file must meet certain requirements:
# - it must contain ONLY syntactically valid python code (any syntax
#   or indentation error that stops the file from running would result
#   in a mark of zero);
# - you MAY NOT use global variables; the function must use only the
#   input provided to it in its arguments.

import numpy as np

def moving_average(seq, wsize):
    avg_results = []
    for i in range(len(seq) - wsize + 1):
        avg_results.append(np.mean(seq[i:i+wsize]))
    return avg_results


def seq_matches(seq1, seq2):
    """
    Return True if two sequences of numbers match with a tolerance of 0.001
    """
    if len(seq1) != len(seq2):
        return False
    for i in range(len(seq1)):
        if abs(seq1[i] - seq2[i]) > 1e-3:
            return False
    return True

def test_moving_average():
    """
    This function runs a number of tests of the moving_average function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    """
    
    assert seq_matches(moving_average((-1, 0, 0, -2, 1), 2), (-0.5, 0.0, -1.0, -0.5))
    assert seq_matches(moving_average([-1, 0, 0, -2, 1], 3), (-0.334, -0.667, -0.334))
    assert seq_matches(moving_average(np.array([-1, 0, 0, -2, 1]), 4), (-0.75, -0.25))
    assert seq_matches(moving_average((0, 1, 2, 0, 2), 2), (0.5, 1.5, 1.0, 1.0))
    assert seq_matches(moving_average((0, 1, 2, 0, 2), 3), (1.0, 1.0, 1.333))
    assert seq_matches(moving_average((0, 1, 2, 0, 2), 4), (0.75, 1.25))

    assert seq_matches(moving_average((-0.4, -0.4, 1.2, -1.6, 1.2), 2), (-0.4, 0.4, -0.2, -0.2))
    assert seq_matches(moving_average((-0.4, -0.4, 1.2, -1.6, 1.2), 3), (0.133, -0.267, 0.266))
    assert seq_matches(moving_average((-0.4, -0.4, 1.2, -1.6, 1.2), 4), (-0.3, 0.1))
    assert seq_matches(moving_average((0.8, 2.0, 0.2, 1.0, 0.4), 2), (1.4, 1.1, 0.6, 0.7))
    assert seq_matches(moving_average((0.8, 2.0, 0.2, 1.0, 0.4), 3), (1.0, 1.066, 0.533))
    assert seq_matches(moving_average((0.8, 2.0, 0.2, 1.0, 0.4), 4), (1.0, 0.9))

    assert seq_matches(moving_average((-1.5, -4.0, -3.0, 3.5, 4.5, 0.0, -3.5, -0.5, 4.0, 0.5), 2), (-2.75, -3.5, 0.25, 4.0, 2.25, -1.75, -2.0, 1.75, 2.25))
    assert seq_matches(moving_average((-1.5, -4.0, -3.0, 3.5, 4.5, 0.0, -3.5, -0.5, 4.0, 0.5), 5), (-0.1, 0.2, 0.3, 0.8, 0.9, 0.1))
    assert seq_matches(moving_average((-1.5, -4.0, -3.0, 3.5, 4.5, 0.0, -3.5, -0.5, 4.0, 0.5), 8), (-0.563, 0.125, 0.687))
    assert seq_matches(moving_average((2.5, -1.0, 1.0, 3.5, -5.0, -0.5, 4.5, -5.0, 5.0, -3.5), 2), (0.75, 0.0, 2.25, -0.75, -2.75, 2.0, -0.25, 0.0, 0.75))
    assert seq_matches(moving_average((2.5, -1.0, 1.0, 3.5, -5.0, -0.5, 4.5, -5.0, 5.0, -3.5), 5), (0.2, -0.4, 0.7, -0.5, -0.2, 0.1))
    assert seq_matches(moving_average((2.5, -1.0, 1.0, 3.5, -5.0, -0.5, 4.5, -5.0, 5.0, -3.5), 8), (0.0, 0.312, 0.0))
    assert seq_matches(moving_average((2.5, -2.0, -2.5, 2.5, -0.5, -2.5, 0.5, -5.0, 4.5, -4.5, 3.0, 3.5, -4.0, 1.0, 5.0, 1.0, -1.0, 2.0, 4.0, -2.0), 2), (0.25, -2.25, 0.0, 1.0, -1.5, -1.0, -2.25, -0.25, 0.0, -0.75, 3.25, -0.25, -1.5, 3.0, 3.0, 0.0, 0.5, 3.0, 1.0))
    assert seq_matches(moving_average((2.5, -2.0, -2.5, 2.5, -0.5, -2.5, 0.5, -5.0, 4.5, -4.5, 3.0, 3.5, -4.0, 1.0, 5.0, 1.0, -1.0, 2.0, 4.0, -2.0), 5), (0.0, -1.0, -0.5, -1.0, -0.6, -1.4, -0.3, 0.3, 0.5, -0.2, 1.7, 1.3, 0.4, 1.6, 2.2, 0.8))
    assert seq_matches(moving_average((2.5, -2.0, -2.5, 2.5, -0.5, -2.5, 0.5, -5.0, 4.5, -4.5, 3.0, 3.5, -4.0, 1.0, 5.0, 1.0, -1.0, 2.0, 4.0, -2.0), 8), (-0.875, -0.625, -0.938, -0.25, -0.125, -0.563, -0.125, 0.437, 1.187, 0.5, 1.312, 1.437, 0.75))
    assert seq_matches(moving_average((2.5, -2.0, -2.5, 2.5, -0.5, -2.5, 0.5, -5.0, 4.5, -4.5, 3.0, 3.5, -4.0, 1.0, 5.0, 1.0, -1.0, 2.0, 4.0, -2.0), 13), (-0.347, -0.462, 0.076, 0.346, 0.076, 0.269, 0.769, 0.576))
    assert seq_matches(moving_average((-2.5, 3.5, 0.0, 3.5, 1.0, -2.5, -4.0, 1.5, -3.5, -3.0, 1.5, 0.0, 1.5, -3.5, -4.0, 3.5, 4.5, 2.5, 0.5, 0.5), 2), (0.5, 1.75, 1.75, 2.25, -0.75, -3.25, -1.25, -1.0, -3.25, -0.75, 0.75, 0.75, -1.0, -3.75, -0.25, 4.0, 3.5, 1.5, 0.5))
    assert seq_matches(moving_average((-2.5, 3.5, 0.0, 3.5, 1.0, -2.5, -4.0, 1.5, -3.5, -3.0, 1.5, 0.0, 1.5, -3.5, -4.0, 3.5, 4.5, 2.5, 0.5, 0.5), 5), (1.1, 1.1, -0.4, -0.1, -1.5, -2.3, -1.5, -0.7, -0.7, -0.7, -0.9, -0.5, 0.4, 0.6, 1.4, 2.3))
    assert seq_matches(moving_average((-2.5, 3.5, 0.0, 3.5, 1.0, -2.5, -4.0, 1.5, -3.5, -3.0, 1.5, 0.0, 1.5, -3.5, -4.0, 3.5, 4.5, 2.5, 0.5, 0.5), 8), (0.062, -0.063, -0.875, -0.688, -1.125, -1.063, -1.188, -1.188, -0.938, 0.062, 0.75, 0.625, 0.687))
    assert seq_matches(moving_average((-2.5, 3.5, 0.0, 3.5, 1.0, -2.5, -4.0, 1.5, -3.5, -3.0, 1.5, 0.0, 1.5, -3.5, -4.0, 3.5, 4.5, 2.5, 0.5, 0.5), 13), (-0.231, -0.308, -0.885, -0.616, -0.539, -0.424, -0.193, 0.153))

    print("all tests passed")


test_moving_average()

