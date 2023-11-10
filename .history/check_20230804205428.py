"""
This program computes how much 1000 dollars have grown to after three years with 5 percent interest rate.
The result is stored in a variable named final_amount.

The correct output is:
the final amount is 1157.63
"""

amount = 1000
years = 3
rate = 5

final_amount = amount * (1 + rate / 100) ** years
print("the final amount is", final_amount)