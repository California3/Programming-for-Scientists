# coding: utf-8
# COMP1730/6730 S1 2023 --- Homework 4
# Submission is due 11:55pm, Saturday the 29th of April, 2023.

# YOUR ANU ID: u7095197
# YOUR NAME: Yangbohan Miao

# You should implement one function stock_trade; you may define
# more functions if this will help you to achieve the functional
# correctness, and to improve the code quality of you program

import math


def stock_trade(stock_price, capital, p):
    '''
    something important is written here...
    '''
    assert all([v > 0 for v in stock_price])
    assert 0.0 <= p <= 1.0
    assert capital >= 0.0
    if len(stock_price) == 0:
        return 0  # no trade

    current_stock = 0
    current_capital = capital * 1.0

    temp = (int) (current_capital * (1-p) / stock_price[0])
    current_stock = temp
    current_capital -= temp * stock_price[0]
    for i in range(1,len(stock_price)):
        if stock_price[i] == stock_price[i-1]:
            continue
        elif stock_price[i] < stock_price[i-1]:
            temp = (int) (current_capital * (1-p) / stock_price[i])
            current_stock += temp
            current_capital -= temp * stock_price[i]
        else:
            temp = (int) (current_stock * (1-p))
            current_stock -= temp
            current_capital += temp * stock_price[i]

    return (float) (current_capital + current_stock * stock_price[-1] - capital)


def test_stock_trade():
    '''
    some typical trading situations but by no means exhaustive
    '''
    assert math.isclose(stock_trade([1, 1, 1, 1, 1], 100, 0.5), 0.0)
    assert math.isclose(stock_trade([100, 50, 50], 10, 0.01), 0.0)
    assert math.isclose(stock_trade([50, 100, 50], 10, 0.01), 0.0)
    assert math.isclose(stock_trade([1, 2, 3, 4, 5], 2, 0.5), 5-1)
    assert math.isclose(stock_trade(tuple(), 100, 0.5), 0.0)
    assert math.isclose(stock_trade([1, 10, 2.0, 5.0], 50, 0.5), 268.0)
    assert math.isclose(stock_trade([1, 10, 2.0, 2.0, 5.0, 5], 50, 0.5), 268.0)
    assert math.isclose(stock_trade([0.01, 0.02], 0.4, 0.5), 0.2)
    print('all tests passed')