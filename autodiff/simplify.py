from typing import Type
import autodiff as ad

class BaseSimplify:
    @staticmethod
    def neg(x: ad.base.BaseOp) -> ad.base.BaseOp:
        return -x

    @staticmethod
    def recip(x: ad.base.BaseOp) -> ad.base.BaseOp:
        return ad.operators.Recip(x)
    
    @staticmethod
    def exp(x: ad.base.BaseOp) -> ad.base.BaseOp:
        return ad.exp(x)
    
    @staticmethod
    def ln(x: ad.base.BaseOp) -> ad.base.BaseOp:
        return ad.ln(x)
    
    @staticmethod
    def lg(x: ad.base.BaseOp) -> ad.base.BaseOp:
        return ad.lg(x)
    
    @staticmethod
    def sqrt(x: ad.base.BaseOp) -> ad.base.BaseOp:
        return ad.sqrt(x)
    
    @staticmethod
    def cbrt(x: ad.base.BaseOp) -> ad.base.BaseOp:
        return ad.cbrt(x)
    
    @staticmethod
    def add(x: ad.base.BaseOp, y: ad.base.BaseOp) -> ad.base.BaseOp:
        return x + y
    
    @staticmethod
    def sub(x: ad.base.BaseOp, y: ad.base.BaseOp) -> ad.base.BaseOp:
        return x - y
    
    @staticmethod
    def mul(x: ad.base.BaseOp, y: ad.base.BaseOp) -> ad.base.BaseOp:
        return x * y
    
    @staticmethod
    def div(x: ad.base.BaseOp, y: ad.base.BaseOp) -> ad.base.BaseOp:
        return x / y

    @staticmethod
    def pow(x: ad.base.BaseOp, y: ad.base.BaseOp) -> ad.base.BaseOp:
        return x ** y
    

    @classmethod
    def _simplify(cls, op: ad.base.BaseOp) -> ad.base.BaseOp:
        if isinstance(op, ad.base.UnaryOp):
            op = type(op)(cls._simplify(op.x))
            return get_method(cls, op)(op.x)
        if isinstance(op, ad.base.BinaryOp):
            op = type(op)(cls._simplify(op.x), cls._simplify(op.y))
            return get_method(cls, op)(op.x, op.y)
        if isinstance(op, ad.base.MultiOp):
            raise NotImplementedError()
            op = type(op)([cls._simplify(var) for var in op.vars])
        return op
        
    @classmethod
    def simplify(cls, op: ad.base.BaseOp) -> ad.base.BaseOp:
        while True:
            new_op = cls._simplify(op)
            if new_op == op:
                return op
            op = new_op
        

def get_method(cls: Type[BaseSimplify], op):
    return {
        ad.operators.Add: cls.add,
        ad.operators.Sub: cls.sub,
        ad.operators.Mul: cls.mul,
        ad.operators.Div: cls.div,
        ad.operators.Pow: cls.pow,

        ad.operators.Neg: cls.neg,
        ad.operators.Recip: cls.recip,

        ad.exp: cls.exp,
        ad.ln: cls.ln,
        ad.lg: cls.lg,
        ad.sqrt: cls.sqrt,
        ad.cbrt: cls.cbrt,
    }[op]


class ConstSimplify(BaseSimplify):
    @staticmethod
    def add(x, y):
        if x == 0:
            return y
        if y == 0:
            return x
        return x + y
    
    @staticmethod
    def sub(x, y):
        if x == 0:
            return -y
        if y == 0:
            return x
        return x - y

    @staticmethod
    def mul(x, y):
        if x == 0 or y == 0:
            return ad.IntConst(0)
        if x == 1:
            return y
        if y == 1:
            return x
        return x * y
    
    @staticmethod
    def div(x, y):
        if x == 0:
            return ad.IntConst(0)
        if y == 1:
            return x
        return x / y
    
    @staticmethod
    def pow(x, y):
        if y == 0 or x == 1:
            return ad.IntConst(1)
        
        if y == 1:
            return x
        
        if x == 0:
            return ad.IntConst(0)
