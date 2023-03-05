from __future__ import annotations
from functools import reduce
import autodiff as ad
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union


class BaseOp(ABC):
    priority: int = -1

    @abstractmethod
    def call(self, vars: Dict[str, float]) -> float:
        pass

    @abstractmethod
    def derivative(self, var: BaseOp) -> BaseOp:
        pass

    def optimize(self) -> BaseOp:
        return self

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    def __neg__(self):
        return ad.operators.Neg(self)

    def __add__(self, other) -> BaseOp:
        return ad.operators.Add(self, other)

    def __sub__(self, other) -> BaseOp:
        return ad.operators.Sub(self, other)

    def __mul__(self, other) -> BaseOp:
        return ad.operators.Mul(self, other)

    def __truediv__(self, other) -> BaseOp:
        return ad.operators.Div(self, other)

    def __pow__(self, other) -> BaseOp:
        return ad.operators.Pow(self, other)

    def __radd__(self, other) -> BaseOp:
        return ad.operators.Add(other, self)

    def __rsub__(self, other) -> BaseOp:
        return ad.operators.Sub(other, self)

    def __rmul__(self, other) -> BaseOp:
        return ad.operators.Mul(other, self)

    def __rtruediv__(self, other) -> BaseOp:
        return ad.operators.Div(other, self)

    def __rpow__(self, other) -> BaseOp:
        return ad.operators.Pow(other, self)


class Variable(BaseOp):
    def __init__(self, name: str):
        self.name = str(name)

    def call(self, vars):
        try:
            return vars[self.name]
        except KeyError:
            raise ValueError(f"Value is not specified for {self.name}")

    def derivative(self, var):
        return IntConst(self == var)
    
    def __str__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, Variable) and self.name == other.name


class Const(BaseOp):
    def __init__(self, value: float):
        self.value = float(value)

    def call(self, vars):
        return self.value

    def derivative(self, var):
        return IntConst(0)
    
    def optimize(self):
        if self.value.is_integer():
            return IntConst(int(self.value))
        return self
    
    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        return (
            isinstance(other, Const)
            and not isinstance(other, IntConst)
            and self.value == other.value
        )


class IntConst(Const):
    def __init__(self, value: int):
        self.value = int(value)

    def call(self, vars):
        return float(self.value)

    def __eq__(self, other):
        return isinstance(other, IntConst) and self.value == other.value


def to_op(obj: Union[BaseOp, str, float, int]) -> BaseOp:
    if isinstance(obj, BaseOp):
        return obj
    if isinstance(obj, str):
        return Variable(obj)
    if isinstance(obj, float):
        return Const(obj)
    if isinstance(obj, int):
        return IntConst(obj)

    raise TypeError("Unsupported type: {type(obj)}")


class UnaryOp(BaseOp):
    op: str = None  # type: ignore

    def __init__(self, x):
        self.x = to_op(x)

    def call(self, vars):
        return self._call(self.x.call(vars))

    def derivative(self, var):
        return self._derivative(self.x) * self.x.derivative(var)
    
    def __str__(self):
        s = str(self.x)
        if not (s.startswith("(") and s.startswith(")")):
            s = f"({s})"
        return f"{self.op}{s}"

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.x == other.x

    @staticmethod
    @abstractmethod
    def _call(x: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def _derivative(x: BaseOp) -> BaseOp:
        pass


class BinaryOp(BaseOp):
    op: str = None  # type: ignore

    def __init__(self, x, y):
        self.x = to_op(x)
        self.y = to_op(y)

    def call(self, vars):
        return self._call(self.x.call(vars), self.y.call(vars))

    def derivative(self, var):
        dx, dy = self._derivative(self.x, self.y)
        return dx * self.x.derivative(var) + dy * self.y.derivative(var)
    
    def __str__(self):
        s1, s2 = str(self.x), str(self.y)
        if isinstance(self.x, BinaryOp):
            s1 = f"({s1})"
        if isinstance(self.y, BinaryOp):
            s2 = f"({s2})"

        return f"{s1} {self.op} {s2}"

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.x == other.x and self.y == other.y

    @staticmethod
    @abstractmethod
    def _call(x: float, y: float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def _derivative(x: BaseOp, y: BaseOp) -> Tuple[BaseOp, BaseOp]:
        pass

class MultiOp(BaseOp):
    op: str = None  # type: ignore

    def __init__(self, *vars):
        self.vars = list(map(to_op, vars))

    def call(self, vars):
        return self._call([var.call(vars) for var in self.vars])

    def derivative(self, var):
        derivatives = self._derivative(self.vars)
        vars = [var_.derivative(var) for var_ in self.vars]

        return sum(map(ad.operators.Mul, derivatives, vars))

    def __str__(self):
        args_str = " {self.op} ".join(map(str, self.vars))
        return f"{self.op}({args_str})"

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.vars == other.vars

    @staticmethod
    @abstractmethod
    def _call(vars: List[float]) -> float:
        pass

    @staticmethod
    @abstractmethod
    def _derivative(vars: List[BaseOp]) -> List[BaseOp]:
        pass
