import math
import builtins

import autodiff as ad

from autodiff import base
from autodiff.base import *

from autodiff import operators

from autodiff import functions
from autodiff.functions import *

from autodiff import trigonometry
from autodiff.trigonometry import *

from autodiff import simplify
from autodiff.simplify import *


def _is_neg(op: Base) -> bool:
    if isinstance(op, (FloatConst, IntConst)):
        return op.value < 0
    return isinstance(op, ad.operators.Neg)


def _abs(op: Base) -> Base:
    if isinstance(op, (FloatConst, IntConst)):
        return type(op)(builtins.abs(op.value)) # type: ignore
    if isinstance(op, ad.operators.Neg):
        return op.op
    return op


def _is_inv(op: Base) -> bool:
    return isinstance(op, ad.operators.Inv)


def _abs_inv(op: Base) -> Base:
    if isinstance(op, ad.operators.Inv):
        return op.op
    return op


def _is_const(op: Base) -> bool:
    return isinstance(op, (ComplexConst, FloatConst, IntConst))

def _is_constant(op: Base) -> bool:
    return isinstance(op, ad.Constant)


def _to_float(value: complex) -> float:
    if math.isclose(value.imag, 0.0):
        return value.real
    return float("nan")
