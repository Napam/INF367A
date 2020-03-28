
def c(x):
    return 1-x

pi = 0.01
th1 = 0.7
th2 = 0.4

def f(nt, nh):
    numerator = (pi * th1**nt * c(th1)**nh)
    denominator = numerator + c(pi) * th2**nt * c(th2)**nh
    return numerator / denominator

print(f(36,64))

exit()
numerator = 0.07203*0.46
denominator = numerator + 0.01536*0.54

print(1-numerator/denominator)