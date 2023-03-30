import cmath

import autodiff as ad


class Sin(ad.Function):
    name = "sin"

    @staticmethod
    def _func_call(x):
        return cmath.sin(x)

    @staticmethod
    def _func_derivative(op):
        return ad.cos(op)


class Cos(ad.Function):
    name = "cos"

    @staticmethod
    def _func_call(x):
        return cmath.cos(x)

    @staticmethod
    def _func_derivative(op):
        return -ad.sin(op)


class Tg(ad.Function):
    name = "tg"

    @staticmethod
    def _func_call(x):
        return cmath.tan(x)

    @staticmethod
    def _func_derivative(op):
        return 1 / (ad.cos(op) ** 2)


class Ctg(ad.Function):
    name = "ctg"

    @staticmethod
    def _func_call(x):
        return 1 / cmath.tan(x)

    @staticmethod
    def _func_derivative(op):
        return -1 / (ad.sin(op) ** 2)


class ArcSin(ad.Function):
    name = "arcsin"

    @staticmethod
    def _func_call(x):
        return cmath.asin(x)

    @staticmethod
    def _func_derivative(op):
        return 1 / ad.sqrt(1 - op**2)


class ArcCos(ad.Function):
    name = "arccos"

    @staticmethod
    def _func_call(x):
        return cmath.acos(x)

    @staticmethod
    def _func_derivative(op):
        return -1 / ad.sqrt(1 - op**2)


class ArcTg(ad.Function):
    name = "arctg"

    @staticmethod
    def _func_call(x):
        return cmath.atan(x)

    @staticmethod
    def _func_derivative(op):
        return 1 / (1 + op**2)


class ArcCtg(ad.Function):
    name = "arcctg"

    @staticmethod
    def _func_call(x):
        return cmath.atan(1 / x)

    @staticmethod
    def _func_derivative(op):
        return -1 / (1 + op**2)


sin = Sin
cos = Cos
tg = Tg
ctg = Ctg


arcsin = ArcSin
arccos = ArcCos
arctg = ArcTg
arcctg = ArcCtg

__all__ = ["sin", "cos", "tg", "ctg", "arcsin", "arccos", "arctg", "arcctg"]
