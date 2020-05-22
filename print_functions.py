import numpy as np
import array_to_latex as a2l
from IPython.display import Latex


def print_matrix(np_array_matrix):
    """
    Parameters:
    np_array_matrix - 2 dimensional ndarray.

    returns IPython Latex object representing the input matrix.
    """

    assert len(
        np_array_matrix.shape) == 2, "Are you rying to print multi dimmensional matrix?"
    latex_code = a2l.to_ltx(
        np_array_matrix, frmt='{:2f}', arraytype='pmatrix', print_out=False)
    return Latex(latex_code)
