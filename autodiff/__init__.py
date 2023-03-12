from autodiff import base
from autodiff import operators
from autodiff import functions

from autodiff.base import Variable, Const, IntConst, optimize
from autodiff.functions import *

from autodiff.simplify import ConstSimplify


nan = float("nan")

def isnan(n: float) -> bool:
    return isinstance(n, float) and (n != n)
