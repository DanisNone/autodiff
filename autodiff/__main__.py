import autodiff as ad

if __name__ == "__main__":
    x = ad.Variable("x")
#    y = -ad.ln(
#        (x**6 + 15*x**4 - 80*x**3 + 27*x**2 - 528*x + 781)
#        * ad.sqrt(x**4 + 10*x**2 - 96*x - 71)
#        - (x**8 + 20*x**6 - 128*x**5 + 54*x**4 - 1408*x**3 + 3124*x**2 + 10001)
#    ) / 8
    y = ad.tg(x).derivative(x).derivative(x).derivative(x)

    print(y)