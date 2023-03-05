from autodiff import base
from autodiff import operators
from autodiff import functions

from autodiff.base import Variable, Const, IntConst
from autodiff.functions import ln, lg, sqrt, cbrt


nan = float("nan")


def isnan(n: float) -> bool:
    return isinstance(n, float) and (n != n)
