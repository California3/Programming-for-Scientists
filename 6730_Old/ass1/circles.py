# COMP1730/6730 Homework 1.

# YOUR ANU ID: u7095197
# YOUR NAME: Yangbohan Miao

# Your task in this homework is to write several (nine, to be precise)
# print function calls to make your program output a circle made
# of simple characters like * and - (and a few others), so called
# "ascii art". What your program prints out should look like this
# (notice that the circle centre is also marked):
#
#
#                               -
#                           *       *
#                         *           *
#                       /               \
#                      (        x        )
#                       \               /
#                         *           *
#                           *      *
#                               -
#
# The horizontal coordinate of the circle centre must be defined as a variable
# before print calls (for example x=10), such that by changing its value the
# circle will shift left of right. Make your program to print three
# such circles at different horizontal positions, one under another.
# Think how repetitions in calling print functions can be avoided
# (Hint: by defining a function).
def circle(x):
    print((x-1)*" "+"-")
    print((x-5)*" "+"*       *")
    print((x-7)*" "+"*           *")
    print((x-9)*" "+"/               \\")
    print((x-10)*" "+"(        x        )")
    print((x-9)*" "+"\\               /")
    print((x-7)*" "+"*           *")
    print((x-5)*" "+"*       *")
    print((x-1)*" "+"-")

x=10
circle(x)
x=30
circle(x)
x=50
circle(x)
