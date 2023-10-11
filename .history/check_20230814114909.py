def mystery( n ):
    k = 0
    while n > 0:
        if n < 10:
            return k
        n = n // 10
        k = k + 1
    if k > 0:
        return k

# Path: check.py
mystery(0)