## COMP1730/6730 - Homework 4

# Your ANU ID: u7100771
# Your NAME: Guangming Zeng
# I declare that this submission is my own work
# [ref: https://www.anu.edu.au/students/academic-skills/academic-integrity ]

## Implement the function below.
## (The statement "pass" is just a placeholder that does nothing: you
## should replace it.)
## You can define other functions if it helps you decompose the problem
## and write a better organised and/or more readable solution.

def linear_prediction(x, y, x_test):
    # sort in ascending order
    x_sorted = sorted(x)
    x1 = x_sorted[0]
    x2 = x_sorted[1]
    # find near by x1, x2 for x_test
    for i in range(len(x_sorted) -1):
        if x_sorted[i] <= x_test:
            x1 = x_sorted[i]
            x2 = x_sorted[i+1]
        else:
            break
    # if x_test is exactly one of the sample points
    if x_test == x1:
        return y[x.index(x1)]
    if x_test == x2:
        return y[x.index(x2)]

    # find corresponding y1, y2 for x1, x2
    y1 = y[x.index(x1)]
    y2 = y[x.index(x2)]

    # calculate the slope and intercept
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    return a * x_test + b

################################################################################
#               DO NOT MODIFY ANYTHING BELOW THIS POINT
################################################################################

def test_linear_prediction():
    '''
    This function runs a number of tests of the linear_prediction function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    '''

    assert abs(linear_prediction([1.0, 3.0, 5.0], [1.0, 9.0, 25.0], 0.5) - -1.0) < 1e-6
    assert abs(linear_prediction([1.0, 3.0, 5.0], [1.0, 9.0, 25.0], 2.0) - 5.0) < 1e-6
    assert abs(linear_prediction([1.0, 3.0, 5.0], [1.0, 9.0, 25.0], 4.0) - 17.0) < 1e-6
    assert abs(linear_prediction([1.0, 3.0, 5.0], (1.0, 9.0, 25.0), 6.0) - 33.0) < 1e-6
    assert abs(linear_prediction((1.0, 5.0, 3.0), [1.0, 25.0, 9.0], 1.25) - 2.0) < 1e-6
    assert abs(linear_prediction((1.0, 5.0, 3.0), (1.0, 25.0, 9.0), 2.5) - 7.0) < 1e-6

    # test that we get the right answer when x_test is exactly one
    # of the sample points:
    assert abs(linear_prediction([1.0, 3.0, 5.0], [1.0, 9.0, 25.0], 1) - 1.0) < 1e-6
    assert abs(linear_prediction([5.0, 1.0, 3.0], [25.0, 1.0, 9.0], 3) - 9.0) < 1e-6
    assert abs(linear_prediction([3.0, 1.0, 5.0], [9.0, 1.0, 25.0], 5) - 25.0) < 1e-6

    # we should get the same answer also if only the two adjacent
    # sample points are given:
    assert abs(linear_prediction([1.0, 3.0], [1.0, 9.0], 0) - -3) < 1e-6
    assert abs(linear_prediction([3.0, 1.0], [9.0, 1.0], 2.0) - 5.0) < 1e-6
    assert abs(linear_prediction([5.0, 3.0], [25.0, 9.0], 4.0) - 17.0) < 1e-6
    assert abs(linear_prediction([1.0, 3.0], [1.0, 9.0], 4.0) - 13.0) < 1e-6

    print("all tests passed")

import matplotlib.pyplot as plt

def plot_linear_prediction(x, y, x_tests):
    """
    This function visualizes linear_prediction results.
    It takes multiple x_test values as a sequence x_tests and
    data points specified in sequences x and y.
    Args:
        x: sequence of x-values
        y: sequence of corresponding y-values
        x_tests: sequence of testing x-values
    """
    y_tests = [ linear_prediction(x, y, x_test) for x_test in x_tests ]
    xlim_min = min(min(x), min(x_tests))-1
    ylim_min = min(min(y), min(y_tests))-3
    plt.xlim(xlim_min, max(max(x), max(x_tests))+1)
    plt.ylim(ylim_min, max(max(y), max(y_tests))+3)
    plt.plot(x, y, marker = "o", color = "black")
    for x_test, y_test in zip(x_tests, y_tests):
        plt.plot([xlim_min, x_test, x_test], [y_test, y_test, ylim_min], 
            linestyle = 'dashed')
    plt.show()

# test_linear_prediction()