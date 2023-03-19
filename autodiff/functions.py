import autodiff as ad
import math


class Exp(ad.base.UnaryOp):
    op = "exp"

    @staticmethod
    def _call(x):
        return math.exp(x)

    @staticmethod
    def _derivative(x):
        return ad.exp(x)


class NaturalLog(ad.base.UnaryOp):
    op = "ln"

    @staticmethod
    def _call(x):
        if x <= 0:
            return ad.nan
        return math.log(x)

    @staticmethod
    def _derivative(x):
        return 1 / x


class Log10(ad.base.UnaryOp):
    op = "lg"

    @staticmethod
    def _call(x):
        if x <= 0:
            return ad.nan
        return math.log10(x)

    @staticmethod
    def _derivative(x):
        return 1 / (x * ad.ln(10))


class Sqrt(ad.base.UnaryOp):
    op = "sqrt"

    @staticmethod
    def _call(x):
        if x < 0:
            return ad.nan
        return math.sqrt(x)

    @staticmethod
    def _derivative(x):
        return 1 / (2 * ad.sqrt(x))


class Cbrt(ad.base.UnaryOp):
    op = "cbrt"

    @staticmethod
    def _call(x):
        return abs(x) ** (1 / 3) * (1 if x > 0 else -1)

    @staticmethod
    def _derivative(x):
        return 1 / (3 * ad.cbrt(x) ** 2)


exp = Exp
ln = NaturalLog
lg = Log10
sqrt = Sqrt
cbrt = Cbrt


__all__ = ["exp", "ln", "lg", "sqrt", "cbrt"]
