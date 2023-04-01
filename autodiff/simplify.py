from typing import List, Tuple
import math

import autodiff as ad


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Simplifier(metaclass=Singleton):
    def __call__(self, op: ad.Base) -> ad.Base:
        while True:
            new_op = self.simplify(op)
            if new_op == op:
                return op
            op = new_op
            
    
    def simplify(self, op):
        ops = op.get_operands()
        if ops:
            for i in range(len(ops)):
                ops[i] = self.simplify(ops[i])
            op = type(op)(*ops)
        method = self.get_method(op.name)
        if method is None:
            return op
        new_op = method(*ops)
        if new_op is None:
            return op
        return new_op

    def get_method(self, name: str):
        return getattr(self, name, None)

    def simplify_one_pair(self, ops: List[ad.Base], name: str):
        method = self.get_method(name)
        if method is None:
            return ops

        while True:
            flag = False

            for i in range(len(ops)):
                if flag:
                    break

                for j in range(i + 1, len(ops)):
                    op = method(ops[i], ops[j])
                    if op is not None:
                        del ops[i], ops[j - 1]
                        ops.append(op)
                        flag = True
                        break
            if not flag:
                return ops

    def simplify_two_pair(self, ops1: List[ad.Base], ops2: List[ad.Base], name: str):
        method = self.get_method(name)
        if method is None:
            return ops1, ops2

        while True:
            flag = False

            for i in range(len(ops1)):
                if flag:
                    break

                for j in range(len(ops2)):
                    op = method(ops1[i], ops2[j])
                    if op is not None:
                        del ops1[i], ops2[j]
                        ops1.append(op)
                        flag = True
                        break
            if not flag:
                return ops1, ops2

    def multiadd(self, *op):
        op = ad.operators.Add(*op).flatten()
        ops = op.get_operands()
        if len(ops) == 0:
            return ad.IntConst(0)
        if len(ops) == 1:
            return ops[0]

        positive = [op for op in ops if not ad._is_neg(op)]
        negative = [ad._abs(op) for op in ops if ad._is_neg(op)]

        positive = self.simplify_one_pair(positive, "add")
        negative = self.simplify_one_pair(negative, "add")

        positive, negative = self.simplify_two_pair(positive, negative, "sub")
        return ad.operators.Add(*positive, *map(ad.operators.Neg, negative))

    def multimul(self, *op):
        op = ad.operators.Mul(*op).flatten()
        ops = op.get_operands()
        if len(ops) == 0:
            return ad.IntConst(1)
        if len(ops) == 1:
            return ops[0]

        numerator = [op for op in ops if not ad._is_inv(op)]
        denominator = [ad._abs_inv(op) for op in ops if ad._is_inv(op)]

        numerator = self.simplify_one_pair(numerator, "mul")
        denominator = self.simplify_one_pair(denominator, "mul")

        numerator, denominator = self.simplify_two_pair(numerator, denominator, "div")
        return ad.operators.Mul(*numerator, *map(ad.operators.Inv, denominator))
        

class BaseSimp(Simplifier):
    def neg(self, op):
        if ad._is_const(op):
            return ad.Const(-op.value)
        if ad._is_neg(op):
            return ad._abs(op)
    
    def add(self, op1, op2):
        if ad._is_const(op1) and ad._is_const(op2):
            return ad.Const(op1.value + op2.value)

        if op1 == 0:
            return op2
        if op2 == 0:
            return op1
        if op1 == op2:
            return 2 * op1

    def sub(self, op1, op2):
        if ad._is_const(op1) and ad._is_const(op2):
            return ad.Const(op1.value - op2.value)
        if op1 == op2:
            return ad.IntConst(0)
        if op1 == 0:
            return -op2
        if op2 == 0:
            return op1

    def mul(self, op1, op2):
        if ad._is_const(op1) and ad._is_const(op2):
            return ad.Const(op1.value * op2.value)
        if ad._is_neg(op1):
            return -(ad._abs(op1) * op2)
        if ad._is_neg(op2):
            return -(op1 * ad._abs(op2))
        
        if op1 == 0 or op2 == 0:
            return ad.IntConst(0)
        if op1 == 1:
            return op2
        if op2 == 1:
            return op1
        
        base1, power1 = _get_power(op1)
        base2, power2 = _get_power(op2)
        
        if base1 == base2:
            return base1 ** (power1 + power2)
        if power1 == power2 and power1 != 1:
            return (base1 * base2) ** (power1)

    def div(self, op1, op2):
        if op2 == 0:
            return None

        if isinstance(op1, ad.IntConst) and isinstance(op2, ad.IntConst):
            x, y = op1.value, op2.value
            n = math.gcd(x, y)
            if n != 1:
                return ad.Const(x // n) / ad.Const(y // n)
        if ad._is_neg(op1):
            return -(ad._abs(op1) / op2)
        if ad._is_neg(op2):
            return -(op1 / ad._abs(op2))
        if op2 == 1:
            return op1
        if op1 == 0:
            return ad.IntConst(0)
        if op1 == op2:
            return ad.IntConst(1)
        
        
        
        base1, power1 = _get_power(op1)
        base2, power2 = _get_power(op2)
        
        
        if base1 == base2:
            return base1 ** (power1 - power2)
        if power1 == power2 and power1 != 1:
            return (base1 / base2) ** (power1)
    def pow(self, op1, op2):
        if op2 == 0:
            return ad.IntConst(1)
        if op2 == 1 or op1 == 1:
            return op1
        if op1 == 0 and ad._is_const(op2):
            n = complex(op2.value)
            if n.real > 0:
                return op1
        if ad._is_neg(op2):
            return 1 / (op1 ** ad._abs(op2))
        if isinstance(op1, ad.operators.Pow):
            return op1.base ** (op1.power * op2)

    def multiadd(self, *ops):
        powers = []
        for op in ops:
            if ad._is_neg(op):
                op = ad._abs(op)
            if isinstance(op, ad.operators.Mul):
                operands = op.get_operands()
                power = -1

                for i in range(len(operands)):
                    _, power_ = _get_power(operands[i])
                    power_ = power_.value if isinstance(power_, (ad.FloatConst, ad.IntConst)) else 0
                    power = max(power, power_)
                powers.append(power)
            else:
                _, power_ = _get_power(op)
                powers.append(power_.value if isinstance(power_, (ad.FloatConst, ad.IntConst)) else 0)
        
        ops = [b for a, b in sorted(zip(powers, ops), key=lambda el:-el[0])]
        return super(BaseSimp, self).multiadd(*ops)
    
    def multimul(self, *ops):
        ops = list(ops)
        index = None
        for i in range(len(ops)):
            if ad._is_const(ops[i]):
                index = i
                break
        if index:
            ops.insert(0, ops.pop(index))
        return super(BaseSimp, self).multimul(*ops)
    
    def ln(self, op):
        if ad._is_constant(op) and op.const_name == "e":
            return ad.IntConst(1)
        if isinstance(op, ad.operators.Mul):
            ops = op.get_operands()
            index, power_ = None, None
            for i in range(len(ops)):
                base, power = _get_power(ops[i])
                if ad._is_constant(base) and base.const_name == "e": # type: ignore
                    index = i
                    power_ = power
            if index:
                del ops[index]
                return ad.operators.Mul(*ops) + power_
           
def _get_power(op: ad.Base) -> Tuple[ad.Base, ad.Base]:
    if isinstance(op, ad.operators.Pow):
        return op.base, op.power
    return op, ad.IntConst(1)

basesimp = BaseSimp()

__all__ = ["basesimp"]
