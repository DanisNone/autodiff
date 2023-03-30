from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, List
import functools
import math

import autodiff as ad


class Base(ABC):
    priority: int = -1
    name: str = None # type: ignore
    
    @abstractmethod
    def _call(self, vars: Dict[str, Union[complex, float, int]]) -> complex:
        pass

    @abstractmethod
    def _derivative(self, var: Variable) -> Base:
        pass

    @abstractmethod
    def get_operands(self) -> List[Base]:
        pass

    @abstractmethod
    def get_variables(self) -> set[Variable]:
        pass
    
    @abstractmethod
    def copy(self) -> Base:
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass
    
    def call(
        self, vars: Dict[str, Union[complex, float, int]] = {}, **kwargs
    ) -> complex:
        vars = {**vars, **kwargs}
        try:
            return complex(self._call(vars))
        except (ValueError, ArithmeticError):
            return float("nan")

    def fcall(
        self, vars: Dict[str, Union[complex, float, int]] = {}, **kwargs
    ) -> float:
        return ad._to_float(self.call(vars, **kwargs))
        

    def derivative(self, *vars: Union[Tuple[Variable, int], Variable]) -> Base:
        for var in vars:
            if isinstance(var, tuple):
                var, k = var
            else:
                k = 1
            if not isinstance(var, Variable):
                raise TypeError(f"var must be Variable, not {type(var)}")
            for i in range(k):
                self = self._derivative(var).simplify()
        return self
    
    def simplify(self):
        return ad.simplify.basesimp(self)
    
    def __neg__(self) -> Base:
        return ad.operators.Neg(self)

    def __add__(self, other) -> Base:
        return ad.operators.Add(self, other)

    def __radd__(self, other) -> Base:
        return ad.operators.Add(other, self)

    def __sub__(self, other) -> Base:
        return ad.operators.Add(self, -other)

    def __rsub__(self, other) -> Base:
        return ad.operators.Add(other, -self)

    def __mul__(self, other) -> Base:
        return ad.operators.Mul(self, other)

    def __rmul__(self, other) -> Base:
        return ad.operators.Mul(other, self)

    def __truediv__(self, other) -> Base:
        return ad.operators.Mul(self, ad.operators.Inv(other))

    def __rtruediv__(self, other) -> Base:
        return ad.operators.Mul(other, ad.operators.Inv(self))
    
    def __pow__(self, other) -> Base:
        return ad.operators.Pow(self, other)
    def __rpow__(self, other) -> Base:
        return ad.operators.Pow(other, self)
        


class Variable(Base):
    name = "variable"
    def __init__(self, name: str):
        self.var_name = str(name)

    def _call(self, vars):
        try:
            return complex(vars[self.name])
        except KeyError:
            raise ValueError(f"unknown variable: {self.var_name}")

    def _derivative(self, var):
        return Const(self == var)

    def get_operands(self):
        return []

    def get_variables(self):
        return {self}
    
    def copy(self):
        return Variable(self.var_name)
    
    def __str__(self):
        return self.var_name

    def __eq__(self, other):
        if isinstance(other, str):
            other = Variable(other)
        return isinstance(other, Variable) and self.var_name == other.var_name

    def __hash__(self):
        return hash(self.var_name)


class ComplexConst(Base):
    name = "complexconst"
    def __init__(self, value: complex):
        self.value = complex(value)

    def _call(self, vars):
        return self.value

    def _derivative(self, var):
        return Const(0)

    def get_operands(self):
        return []

    def get_variables(self):
        return {}
    
    def copy(self):
        return ComplexConst(self.value)

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, complex):
            other = ComplexConst(other)
        return isinstance(other, ComplexConst) and self.value == other.value


class FloatConst(Base):
    name = "floatconst"
    def __init__(self, value: float):
        self.value = float(value)

    def _call(self, vars):
        return self.value

    def _derivative(self, var):
        return Const(0)

    def get_operands(self):
        return []

    def get_variables(self):
        return {}
    
    def copy(self):
        return FloatConst(self.value)

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, float):
            other = FloatConst(other)
        return isinstance(other, FloatConst) and self.value == other.value


class IntConst(Base):
    name = "intconst"
    def __init__(self, value: int):
        self.value = int(value)

    def _call(self, vars):
        return self.value

    def _derivative(self, var):
        return Const(0)

    def get_operands(self):
        return []

    def get_variables(self):
        return {}
    
    def copy(self):
        return IntConst(self.value)

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, int):
            other = IntConst(other)
        return isinstance(other, IntConst) and self.value == other.value


class Constant(Base):
    name = "constant"
    
    def __init__(self, name: str, value: float):
        self.const_name = name
        self.value = float(value)
    
    def _call(self, vars):
        return self.value

    def _derivative(self, var):
        return Const(0)

    def get_operands(self):
        return []

    def get_variables(self):
        return {}
    
    def copy(self):
        return Constant(self.const_name, self.value)

    def __str__(self):
        return self.const_name

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return False
        return self.const_name == other.const_name and self.value == other.value

 
def Const(value: Union[complex, float, int]) -> Base:
    if isinstance(value, complex):
        return ComplexConst(value)
    if isinstance(value, float):
        return FloatConst(value)
    if isinstance(value, int):
        return IntConst(value)
    raise TypeError(f"Type {type(value)} cannot be converted to a constant")


def to_op(obj: Union[complex, float, int, str, Base]) -> Base:
    if isinstance(obj, (complex, float, int)):
        return Const(obj)
    if isinstance(obj, str):
        return Variable(obj)
    if isinstance(obj, Base):
        return obj
    raise TypeError(f"Type {type(obj)} cannot be converted to a Base class")



e = Constant("e", 2.718281828459045)

__all__ = [
    "Base",
    "Variable",
    "Const",
    "ComplexConst",
    "FloatConst",
    "IntConst",
    "Constant",
    "to_op",
    
    "e",
]
