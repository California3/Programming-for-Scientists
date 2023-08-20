## COMP1730/6730 Homework 3

# Your ANU ID: u7100771
# Your NAME: Guangming Zeng
# I declare that this submission is my own work
# [ref: https://www.anu.edu.au/students/academic-skills/academic-integrity ]


## You should implement the functions `collatz_step` and `collatz_stopping_time` 
## below. You can define new function(s) if it helps you decompose the problem
## into smaller problems.

def collatz_step(n):
    # If the input number is an even number, the function returns half the number.
    # If the input is an odd number, the function returns three times the input number plus one.
    if n % 2 == 0:
        return n // 2
    else:
        return n * 3 + 1


def collatz_stopping_time(n):
    cnt = 0
    while n != 1:
        n = collatz_step(n)
        cnt += 1
    return cnt

################################################################################
#               DO NOT MODIFY ANYTHING BELOW THIS POINT
################################################################################    

def test_collatz_step():
    '''
    This function runs a number of tests of the collatz step function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    assert type(collatz_step(4)) == int, "Test failed!"
    assert type(collatz_step(5)) == int, "Test failed!"
    assert collatz_step(4) == 2, "Test failed!"
    assert collatz_step(5) == 16 , "Test failed!"
    print("all tests passed")

def test_collatz_stoppping_time():
    '''
    This function runs a number of tests of the collatz_stopping_time function.
    If it works ok, you will just see the output ("all tests passed") at
    the end when you call this function; if some test fails, there will
    be an error message.
    NOTE: passing all tests does not automatically mean that your code is correct
    because this function only tests a limited number of test cases.
    '''
    assert collatz_stopping_time(27)==111, "Test Failed!"
    assert collatz_stopping_time(1)==0, "Test Failed!"
    assert collatz_stopping_time(2)==1, "Test Failed!"
    assert collatz_stopping_time(2048)==11, "Test Failed!"
    print("all tests passed")


# test_collatz_step()
# test_collatz_stoppping_time()