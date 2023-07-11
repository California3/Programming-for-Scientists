# COMP1730/6730 Homework 1.

# YOUR ANU ID: u7100771
# YOUR NAME: Guangming Zeng

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

circles_info = ["(        x        )",
                " \               /",
                "   *           *",
                "     *       *",
                "         -"]

def print_circle(space_cnt = 0):
    max_cnt = len(circles_info)

    # print in reverse order
    for i in range(max_cnt-1, -1, -1):
        current_line = circles_info[i]
        # replace '\' with 'BOTTOM_TOP'
        current_line = current_line.replace('\\', 'BOTTOM_TOP')
        # replace '/' with 'TOP_BOTTOM'
        current_line = current_line.replace('/', 'TOP_BOTTOM')

        # replace 'BOTTOM_TOP' with '/' if exists 'BOTTOM_TOP'
        if 'BOTTOM_TOP' in current_line:
            current_line = current_line.replace('BOTTOM_TOP', '/')
        # replace 'TOP_BOTTOM' with '\' if exists 'TOP_BOTTOM'
        if 'TOP_BOTTOM' in current_line:
            current_line = current_line.replace('TOP_BOTTOM', '\\')
        print(" " * space_cnt + current_line)

    # print in normal order starting from 1
    for i in range(1, max_cnt):
        print(" " * space_cnt + circles_info[i])


print_circle(4)

# execute in command line
if __name__ == '__main__':
    input_str = "0"
    while input_str.isdigit():
        # get input from command line
        input_str = input("Please input a number: ")

        if not input_str.isdigit():
            break

        # convert input to int
        input_num = int(input_str)
        # print circle
        print_circle(input_num)