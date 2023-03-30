import autodiff as ad


class Neg(ad.Base):
    priority = 1
    name = "neg"

    def __init__(self, op: ad.Base):
        self.op = ad.to_op(op)

    def _call(self, vars):
        return -self.op._call(vars)

    def _derivative(self, var):
        return -self.op._derivative(var)

    def get_operands(self):
        return [self.op]

    def get_variables(self):
        return [self.op.get_variables()]

    def copy(self):
        return Neg(self.op.copy())

    def __str__(self):
        if self.op.priority == -1:
            return f"-{self.op}"
        return f"-({self.op})"

    def __eq__(self, other):
        return isinstance(other, Neg) and self.op == other.op


class Inv(ad.Base):
    name = "inv"

    def __init__(self, op: ad.Base):
        self.op = ad.to_op(op)

    def _call(self, vars):
        return 1 / self.op._call(vars)

    def _derivative(self, var):
        return -self.op._derivative(var) / self.op**2

    def get_operands(self):
        return [self.op]

    def get_variables(self):
        return [self.op.get_variables()]

    def copy(self):
        return Inv(self.op.copy())

    def __str__(self):
        if self.op.priority == -1:
            return f"1 / {self.op}"
        return f"1 / ({self.op})"

    def __eq__(self, other):
        return isinstance(other, Inv) and self.op == other.op


class Add(ad.Base):
    priority = 1
    name = "multiadd"

    def __init__(self, *args: ad.Base):
        self.ops = list(map(ad.to_op, args))

    def _call(self, vars):
        return sum(op._call(vars) for op in self.ops)

    def _derivative(self, var):
        return sum(op._derivative(var) for op in self.ops)

    def get_operands(self):
        return self.ops.copy()

    def get_variables(self):
        vars = set()
        for op in self.ops:
            vars |= op.get_variables()
        return vars

    def flatten(self):
        ops = []
        for op in self.ops:
            if isinstance(op, Add):
                ops.extend(op.flatten().ops)
            elif isinstance(op, Neg) and isinstance(op.op, Add):
                ops.extend(map(Neg, op.op.flatten().ops))
            else:
                ops.append(op)
        return Add(*ops)

    def copy(self):
        return Add(*(op.copy() for op in self.ops))

    def __str__(self):
        if not self.ops:
            return "0"

        arr = []
        for op in self.ops:
            arr.append("-" if ad._is_neg(op) else "+")
            op = ad._abs(op)

            if op.priority == -1 or op.priority > self.priority:
                arr.append(f"{op}")
            else:
                arr.append(f"({op})")

        if arr[0] == "-":
            arr[1] = "-" + arr[1]
        del arr[0]
        return " ".join(arr)

    def __eq__(self, other):
        return isinstance(other, Add) and self.ops == other.ops


class Mul(ad.Base):
    priority = 2
    name = "multimul"

    def __init__(self, *args: ad.Base):
        self.ops = list(map(ad.to_op, args))

    def _call(self, vars):
        res = 1.0
        for op in self.ops:
            res *= op._call(vars)
        return res

    def _derivative(self, var):
        result = ad.Const(0)
        for i in range(len(self.ops)):
            result += Mul(*self.ops[:i], *self.ops[i + 1 :]) * self.ops[i]._derivative(
                var
            )
        return result

    def get_operands(self):
        return self.ops.copy()

    def get_variables(self):
        vars = set()
        for op in self.ops:
            vars |= op.get_variables()
        return vars

    def flatten(self):
        ops = []
        for op in self.ops:
            if isinstance(op, Mul):
                ops.extend(op.flatten().ops)
            elif isinstance(op, Inv) and isinstance(op.op, Mul):
                ops.extend(map(Inv, op.op.flatten().ops))
            else:
                ops.append(op)
        return Mul(*ops)

    def copy(self):
        return Mul(*(op.copy() for op in self.ops))

    def __str__(self):
        if not self.ops:
            return "1"

        arr = []
        for op in self.ops:
            arr.append("/" if ad._is_inv(op) else "*")
            op = ad._abs_inv(op)

            if op.priority == -1 or op.priority > self.priority:
                arr.append(f"{op}")
            else:
                arr.append(f"({op})")

        if arr[0] == "/":
            arr.insert(0, "1")
        else:
            del arr[0]
        return " ".join(arr)

    def __eq__(self, other):
        return isinstance(other, Mul) and self.ops == other.ops


class Pow(ad.Base):
    priority = 3
    name = "pow"

    def __init__(self, base, power):
        self.base = ad.to_op(base)
        self.power = ad.to_op(power)

    def _call(self, vars):
        return self.base._call(vars) ** self.power._call(vars)

    def _derivative(self, var):
        # (f(x)**g(x))' = f(x)**(g(x) - 1) * (g(x)*f'(x) + f(x)*ln(x)*g'(x))

        l1 = self.base ** (self.power - 1)
        l21 = self.power * self.base._derivative(var)
        l22 = self.base * ad.ln(self.base) * self.power._derivative(var)
        return l1 * (l21 + l22)

    def get_operands(self):
        return [self.base, self.power]

    def get_variables(self):
        return {*self.base.get_variables(), *self.power.get_variables()}

    def copy(self):
        return Pow(self.base.copy(), self.power.copy())

    def __str__(self):
        s1, s2 = str(self.base), str(self.power)
        if self.base.priority != -1:
            s1 = f"({s1})"
        if self.power.priority != -1:
            s2 = f"({s2})"
        return f"{s1} ** {s2}"

    def __eq__(self, other):
        if not isinstance(other, Pow):
            return False
        return self.base == other.base and self.power == other.power
