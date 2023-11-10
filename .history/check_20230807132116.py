
def funA(x):
    print("A", end=' ')
    return 2*x

def funB( y ):
    print("B", end=' ')
    return funA( y ) + 1

result = funB(2 + funA(1))
print(result)