import autodiff as ad

if __name__ == "__main__":
    x = ad.Variable("x")
    y = ad.Variable("y")
    z = x + x - y + x - x
    print(z)
    print(z.optimize())