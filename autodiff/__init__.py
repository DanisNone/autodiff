import autodiff as ad

from autodiff import base
from autodiff import operators


from autodiff.base import Variable, Const, IntConst
from autodiff.functions import *
from autodiff.trigonometry import *


nan = float("nan")

def isnan(n: float) -> bool:
    return isinstance(n, float) and (n != n)


def _isneg(x: ad.base.BaseOp) -> bool:
    if isinstance(x, ad.Const):
        return x.value < 0
    return isinstance(x, ad.operators.Neg)

def _isinv(x: ad.base.BaseOp) -> bool:
    return isinstance(x, ad.operators.Inv)

def _abs(x: ad.base.BaseOp) -> ad.base.BaseOp:
    if isinstance(x, ad.Const):
        return type(x)(abs(x.value))
    if isinstance(x, ad.operators.Neg):
        return x.x
    return x

def _abs_inv(x: ad.base.BaseOp) -> ad.base.BaseOp:
    if isinstance(x, ad.operators.Inv):
        return x.x
    return x

def _is_const(x: ad.base.BaseOp) -> bool:
    return isinstance(x, ad.Const)