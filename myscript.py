import math

def slow(n):
    s = 0
    for i in range(n):
        s += math.sqrt(i)
    return s

a = slow(3_000_000)
b = slow(3_000_001)
total = a + b
print("total =", total)