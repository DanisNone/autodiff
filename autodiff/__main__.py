import autodiff as ad

if __name__ == "__main__":
    x = ad.Variable("x")
    y = ad.Variable("y")
    z = -x + y*5
    print(z)
    print(ad.optimize(z))
    