import numpy as np
import copy


def make_donut():
    A = np.zeros((6,6))
    A[:,0:3] = 1
    B = copy.deepcopy(A)
    return A, B

A, B = make_donut()
print(A)
print()
print(B)
print()

def func(A,B, n):
    rr = np.array(range(6))
    cc = np.array([n]*6)
    A[rr, cc] = 1
    B[rr, cc] = 2

func(A,B,5)
func(A,B,3)

print(A)
print()
print(B)
print()