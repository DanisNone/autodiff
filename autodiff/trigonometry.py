import math

import autodiff as ad



class Sin(ad.base.UnaryOp):
    op = "sin"

    @staticmethod
    def _call(x):
        return math.sin(x)

    @staticmethod
    def _derivative(x):
        return ad.cos(x)


class Cos(ad.base.UnaryOp):
    op = "cos"

    @staticmethod
    def _call(x):
        return math.cos(x)

    @staticmethod
    def _derivative(x):
        return -ad.sin(x)


class Tg(ad.base.UnaryOp):
    op = "tg"

    @staticmethod
    def _call(x):
        return math.tan(x)

    @staticmethod
    def _derivative(x):
        return 1 / (ad.cos(x) ** 2)

class Ctg(ad.base.UnaryOp):
    op = "ctg"

    @staticmethod
    def _call(x):
        return 1 / math.tan(x)

    @staticmethod
    def _derivative(x):
        return -1 / (ad.sin(x) ** 2)

class ArcSin(ad.base.UnaryOp):
    op = "arcsin"

    @staticmethod
    def _call(x):
        if x < -1 or x > 1:
            return ad.nan
        return math.asin(x)

    @staticmethod
    def _derivative(x):
        return 1 / ad.sqrt(1 - x ** 2)


class ArcCos(ad.base.UnaryOp):
    op = "arccos"

    @staticmethod
    def _call(x):
        if x < -1 or x > 1:
            return ad.nan
        return math.acos(x)

    @staticmethod
    def _derivative(x):
        return -1 / ad.sqrt(1 - x ** 2)


class ArcTg(ad.base.UnaryOp):
    op = "arctg"

    @staticmethod
    def _call(x):
        return math.atan(x)

    @staticmethod
    def _derivative(x):
        return 1 / (1 + x ** 2)



class ArcCtg(ad.base.UnaryOp):
    op = "arcctg"

    @staticmethod
    def _call(x):
        return math.atan(1 / x)

    @staticmethod
    def _derivative(x):
        return -1 / (1 + x ** 2)
    


sin = Sin
cos = Cos
tg = Tg
ctg = Ctg


arcsin = ArcSin
arccos = ArcCos
arctg = ArcTg
arcctg = ArcCtg

__all__ = ["sin", "cos", "tg", "ctg", "arcsin","arccos", "arctg", "arcctg"]