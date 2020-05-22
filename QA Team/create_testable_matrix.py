# %%
import numpy as np
from numpy import matrix


# %%
def create_testable_matrix(p, j):
    a = p.I @ j @ p
    return a
