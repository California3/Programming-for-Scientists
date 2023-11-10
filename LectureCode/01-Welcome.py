# Home loan repayment.

# Wrong One.
P = 500000
r = 0.02
n = 30  # year
A = P * r * (1 + r) ** n / ((1 + r) ** n - 1)
print(A / 12)  # monthly repayment

# Correct One.
P = 500000
r = 0.02 / 12
n = 30 * 12 # month
A = P * r * (1 + r) ** n / ((1 + r) ** n - 1)
print(A)  # monthly repayment