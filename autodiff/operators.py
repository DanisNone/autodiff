import math
import autodiff as ad


class Neg(ad.base.UnaryOp):
    op = "-"
    priority = 0

    @staticmethod
    def _call(x):
        return -x

    @staticmethod
    def _derivative(x):
        return ad.IntConst(-1)


class Inv(ad.base.UnaryOp):
    priority = 0

    @staticmethod
    def _call(x):
        if x == 0:
            return float("nan")
        return 1 / x

    @staticmethod
    def _derivative(x):
        return -1 / (x * x)

    def __str__(self):
        return f"1/({self.x})"


class Pow(ad.base.BinaryOp):
    op = "**"
    priority = 3

    @staticmethod
    def _call(x, y):
        if not y.is_integer() and x < 0:
            return ad.nan
        return x**y

    @staticmethod
    def _derivative(x, y):
        return y * x ** (y - 1), x**y * ad.ln(x)


class MultiAdd(ad.base.MultiOp):
    op = "+"
    priority = 1

    @staticmethod
    def _call(vars):
        return sum(vars)

    @staticmethod
    def _derivative(vars):
        return [ad.IntConst(1) for _ in vars]

    def __str__(self):
        if not self.vars:
            return "0"
        res = []
        for var in self.vars:
            res.append("-" if ad._isneg(var) else "+")
            
            var = ad._abs(var)
            if var.priority == -1 or var.priority > self.priority:
                res.append(f"{var}")
            else:
                res.append(f"({var})")
        
        if ad._isinv(self.vars[0]):
            res[1] = "-" + res[1]
        del res[0]
        return f" ".join(res)

class MultiMul(ad.base.MultiOp):
    op = "*"
    priority = 2

    @staticmethod
    def _call(vars):
        return math.prod(vars)

    @staticmethod
    def _derivative(vars):
        return [MultiMul(*vars[:i], *vars[i+1:]) for i in range(len(vars))]
    
    def __str__(self):
        if not self.vars:
            return "1"
        res = []
        for var in self.vars:
            res.append("/" if ad._isinv(var) else "*")
            
            var = ad._abs_inv(var)
            if var.priority == -1 or var.priority > self.priority:
                res.append(f"{var}")
            else:
                res.append(f"({var})")
        
        if ad._isinv(self.vars[0]):
            res.insert(0, "1")
        else:
            del res[0]
        return f" ".join(res)

