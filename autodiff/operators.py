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


class Add(ad.base.BinaryOp):
    op = "+"
    priority = 1

    @staticmethod
    def _call(x, y):
        return x + y

    @staticmethod
    def _derivative(x, y):
        return ad.IntConst(1), ad.IntConst(1)
    
    def optimize(self):
        self = super(Add, self).optimize()
        return MultiAdd([self.x, self.y])


class Sub(ad.base.BinaryOp):
    op = "-"
    priority = 1
    @staticmethod
    def _call(x, y):
        return x - y

    @staticmethod
    def _derivative(x, y):
        return ad.IntConst(1), ad.IntConst(-1)


class Mul(ad.base.BinaryOp):
    op = "*"
    priority = 2

    @staticmethod 
    def _call(x, y):
        return x * y

    @staticmethod
    def _derivative(x, y):
        return y, x

    def optimize(self):
        self = super(Mul, self).optimize()
        return MultiMul([self.x, self.y])


class Div(ad.base.BinaryOp):
    op = "/"
    priority = 2
    @staticmethod
    def _call(x, y):
        if y == 0:
            return ad.nan
        return x / y

    @staticmethod
    def _derivative(x, y):
        return 1 / y, -x / (y**2)


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
    
    def optimize(self):
        self = super(MultiAdd, self).optimize()
        vars = []

        for var in self.vars:
            if isinstance(var, MultiAdd):
                vars.extend(var.vars)
            else:
                vars.append(var)
        return MultiAdd(vars)


class MultiMul(ad.base.MultiOp):
    op = "*"
    priority = 2
    @staticmethod
    def _call(vars):
        return math.prod(vars)
    
    @staticmethod
    def _derivative(vars):
        derivatives = []
        for i in range(len(vars)):
            derivatives.append(MultiMul(vars[:i] + vars[i+1:]))
        return derivatives


def _isneg(x):
    if isinstance(x, ad.Const):
        return x.value < 0
    return isinstance(x, Neg)

def _abs(x):
    if isinstance(x, ad.Const):
        return type(x)(abs(x.value))
    if isinstance(x, Neg):
        return x.x
    return x