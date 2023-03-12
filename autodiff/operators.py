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


class Recip(ad.base.UnaryOp):
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

    def optimize(self):
        self = super(Sub, self).optimize()
        return MultiAdd([self.x, -self.y])


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
    
    def optimize(self):
        self = super(Div, self).optimize()
        return MultiMul([self.x, Recip(self.y)])


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
        self.vars.sort(key=_isneg)

        return self

    def __str__(self):
        res = []
        for var in self.vars:
            res.append("-" if _isneg(var) else "+")
            
            var = _abs(var)
            if var.priority == -1 or var.priority >= self.priority:
                res.append(f"{var}")
            else:
                res.append(f"({var})")
        
        if _isrecip(self.vars[0]):
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
        return [MultiMul(vars[:i] + vars[i+1:]) for i in range(len(vars))]
    
    def optimize(self):
        self = super(MultiMul, self).optimize()
        self.vars.sort(key=_isrecip)

        return self
    
    def __str__(self):
        res = []
        for var in self.vars:
            res.append("/" if _isrecip(var) else "*")
            
            var = _abs_recip(var)
            if var.priority == -1 or var.priority >= self.priority:
                res.append(f"{var}")
            else:
                res.append(f"({var})")
        
        if _isrecip(self.vars[0]):
            res.insert(0, "1")
        else:
            del res[0]
        return f" ".join(res)

def _isneg(x: ad.base.BaseOp) -> bool:
    if isinstance(x, ad.Const):
        return x.value < 0
    return isinstance(x, Neg)

def _isrecip(x: ad.base.BaseOp) -> bool:
    return isinstance(x, Recip)

def _abs(x: ad.base.BaseOp) -> ad.base.BaseOp:
    if isinstance(x, ad.Const):
        return type(x)(abs(x.value))
    if isinstance(x, Neg):
        return x.x
    return x

def _abs_recip(x: ad.base.BaseOp) -> ad.base.BaseOp:
    if isinstance(x, Recip):
        return x.x
    return x