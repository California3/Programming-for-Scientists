def sum_fractions(a, b):
    k = b
    total = 0
    while k >= a:
        total = total + 1/k
        k = k - 1
    return total

# Path: check.py
sum_fractions(0, 3)