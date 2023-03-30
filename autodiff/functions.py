from abc import abstractmethod
import builtins

import math
import cmath

import autodiff as ad


if not hasattr(math, "cbrt"):
    def _cbrt(x: float) -> float:
        return builtins.abs(x)**(1/3) * (-1 if x < 0 else 1)
    math.cbrt = _cbrt
    del _cbrt

class Function(ad.Base):
    def __init__(self, op):
        self.op = ad.to_op(op)

    def _call(self, vars):
        value = self.op._call(vars)
        return self._func_call(value)

    def _derivative(self, var):
        return self._func_derivative(self.op) * self.op._derivative(var)

    def get_operands(self):
        return [self.op]

    def get_variables(self):
        return self.op.get_variables()

    def copy(self):
        return type(self)(self.op.copy())

    def __str__(self):
        return f"{self.name}({self.op})"

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.op == other.op

    @staticmethod
    @abstractmethod
    def _func_call(value: complex) -> complex:
        pass

    @staticmethod
    @abstractmethod
    def _func_derivative(op: ad.Base) -> ad.Base:
        pass



class Exp(Function):
    name = "exp"
    
    @staticmethod
    def _func_call(value):
        return cmath.exp(value)

    @staticmethod
    def _func_derivative(op):
        return ad.exp(op)
     
class NaturalLog(Function):
    name = "ln"

    @staticmethod
    def _func_call(value):
        return math.log(value) # type: ignore

    @staticmethod
    def _func_derivative(op):
        return 1 / op


class Log10(Function):
    name = "lg"

    @staticmethod
    def _func_call(value):
        return math.log10(value)# type: ignore

    @staticmethod
    def _func_derivative(op):
        return 1 / (op * ad.lg(10))


class Sqrt(Function):
    name = "sqrt"

    @staticmethod
    def _func_call(value):
        return math.sqrt(value) # type: ignore

    @staticmethod
    def _func_derivative(op):
        return 1 / (2 * ad.sqrt(op))


class Cbrt(Function):
    name = "cbrt"

    @staticmethod
    def _func_call(value):
        value = ad._to_float(value)
        return abs(value) ** (1 / 3) * (-1 if value < 0 else 1)

    @staticmethod
    def _func_derivative(op):
        return 1 / (3 * ad.cbrt(op) ** 2)

class Abs(Function):
    name = "abs"

    @staticmethod
    def _func_call(value):
        return abs(value)
    
    @staticmethod
    def _func_derivative(op):
        return abs(op) / op


exp = Exp
ln = NaturalLog
lg = Log10

sqrt = Sqrt
cbrt = Cbrt
abs = Abs

__all__ = ["Function", "exp", "ln", "lg", "sqrt", "cbrt", "abs"]
