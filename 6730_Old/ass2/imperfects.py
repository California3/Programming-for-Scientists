# COMP1730/6730 Homework 2

# YOUR ANU ID: u7095197
# YOUR NAME: Yangbohan Miao

# You should implement the functions
#  1. sum_factors
#  2. deficients and
#  3. excessives


def sum_factors(n):
    '''
    Computes and returns the sum of the factors of the positive integer n
    
    Reminder:
      - This function does not need to return the factors themselves, just the sum of the factors
      - 1 is not a factor of 1 itself, but is a factor for every other positive integer n
    '''
    if n == 1:
        return 0
    else:
        sum = 1
        for i in range(2, n):
            if n % i == 0:
                sum += i

    return sum  # 0 must be replaced by the computed value


def deficients(N):
    '''
    Computes and returns the number of deficient positive integers smaller than the positive integer N
    '''
    sum = 0
    for i in range(1, N):
        if sum_factors(i) < i:
            sum += 1
    
    return sum  # 0 must be replaced by the computed value


def excessives(N):
    '''
    Computes and returns the number of excessive positive integers greater than the positive integer N
    '''
    sum = 0
    for i in range(1, N):
        if sum_factors(i) > i:
            sum += 1

    return sum  # 0 must be replaced by the computed value


def test_sum_factors():
    '''
    This function runs a number of tests of the sum_factors function.
    If it works OK, you will see the output ("all tests for sum_factors passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    '''
    assert sum_factors(1) == 0
    assert sum_factors(2) == 1
    assert sum_factors(3) == 1
    assert sum_factors(100) == 117
    
    print("all tests for sum_factors passed")


def test_deficients():
    '''
    This function runs a number of tests of the deficients function.
    If it works OK, you will see the output ("all tests for deficients passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    '''
    assert deficients(1) == 0
    assert deficients(2) == 1
    assert deficients(3) == 2
    assert deficients(100) == 76
    
    print("all tests for deficients passed")



def test_excessives():
    '''
    This function runs a number of tests of the excessives function.
    If it works OK, you will see the output ("all tests for excessives passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    '''
    assert excessives(1) == 0
    assert excessives(2) == 0
    assert excessives(3) == 0
    assert excessives(100) == 21
    
    print("all tests for excessives passed")