import autodiff as ad

if __name__ == "__main__":
    x = ad.Variable("x")
    y = ad.cbrt(x)
    print(y)
    print(y.derivative(x))
