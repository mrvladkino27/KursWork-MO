import math
import numpy as np
import matplotlib.pylab as pl

STOP_R = False 
COUNTER = 0
count_func = 0
H_Derivative = 0.000000000001
DELTA_MOP = 0.01

R_func = 1

def limit_region(x):
    return 1- (x[0]+1)**2 + (x[1])**2



def IncrementCountFunc():
    global count_func
    count_func = count_func+1

def CountToZero():
    global count_func
    count_func = 0

def function(x):
    IncrementCountFunc()
         
    r = limit_region(x) 
    r = R_func*(1/r)
    return (10*(x[0] - x[1])**2 + (x[0] - 1)**2)**(4) + r
    

def plot(points, col):
    
    xs = []
    ys = []
    for i in range(len(points)):
        xs.append(points[i][0])
        ys.append(points[i][1])
        
    pl.grid()
    pl.plot(xs, ys, marker='o', linestyle='--', color=str(col), label='Square')
    


def calcX(x0, grad, lmb):
    #   x0 - gradient*lambda

    return difference(x0, mults(grad, lmb))

def difference(x, y):
    res = []
    for i in range(len(x)):
        res.append(x[i] - y[i])
    return res        

def mults(x, n):
    res = []
    for i in range(len(x)):
        res.append(x[i]*n)
    return res

def gradient(x):
    grad = []
    for i in range(len(x)):
        # grad.append(CentralDerivative(x, i, H_Derivative))
        grad.append(RightDerivative(x, i, H_Derivative))
    return grad 

def LeftDerivative(x, n, h_value):
    h = []
    for i in range(len(x)):
        if i == n:
            h.append(h_value)
        else:
            h.append(0)
    return (function([x[0] + h[0], x[1] + h[1]]) - function([x[0] - h[0], x[1] - h[1]]))/(2*h[n])

def CentralDerivative(x, n, h_value):
    h = []
    for i in range(len(x)):
        if i == n:
            h.append(h_value)
        else:
            h.append(0)
    return (function([x[0] + h[0], x[1] + h[1]]) - function([x[0] - h[0], x[1] - h[1]]))/(2*h[n])

def RightDerivative(x, n, h_value):
    h = []
    for i in range(len(x)):
        if i == n:
            h.append(h_value)
        else:
            h.append(0)
    return (function([x[0] + h[0], x[1] + h[1]]) - function([x[0], x[1]]))/(2*h[n])

def norm(s1):
    normas = 0
    for i in range(len(s1)):
        normas += s1[i]**2
    return math.sqrt(normas)

def calcLambda(x0, grad, eps, lmb, delta):
    
    # return gold(x0, grad, eps, lmb, delta)
    
    return dscPowell(x0, grad, eps, lmb, delta)
    
def gold(x0, grad, eps, lmb, delta):
    
    #      One-dimensional gold search
    
    [a,b] = svenn(x0, grad, lmb, delta)
    l = b - a
    x1 = a + 0.382*l
    x2 = a + 0.618*l
    while l > eps:
        if function(calcX(x0, grad, x1)) < function(calcX(x0, grad, x2)):
            b = x2
            x2 = x1
            l = b - a
            x1 = a + 0.382*l
        else:
            a = x1
            x1 = x2
            l = b - a
            x2 = a + 0.618*l
    # print("gold a: " + str(a))
    # print("gold b: " + str(b)) 
    return (a + b)/2

def svenn(x0, grad, lmb, delta):
    #
    #    One-dimensional Svenn search
    #
    # print ("Svenn stage...")
    f0 = function(calcX(x0, grad, lmb))
    if f0 < function(calcX(x0, grad, lmb+delta)):
        delta = -delta
    x1 = lmb + delta
    f1 = function(calcX(x0, grad, x1))
    while f1 < f0:
        delta *= 2
        lmb = x1
        x1 = lmb + delta
        f0 = f1
        f1 = function(calcX(x0, grad, x1))
    a = lmb + delta/2
    b = lmb - delta/2        
    if a > b:
        temp = b
        b = a
        a = temp     
    #print "svenn a: " + str(a)
    #print "svenn b: " + str(b)    
    return [a , b]

def dsc(x0, grad, lmb, delta):
    svenn_res = svenn(x0, grad, lmb, delta)
    x1 = svenn_res[0]
    x3 = svenn_res[1]
    x2 = (x1 + x3)/2
    f1 = function(calcX(x0, grad, x1))
    f2 = function(calcX(x0, grad, x2))
    f3 = function(calcX(x0, grad, x3))
    xApprox = x2 + ((x3 - x2) * (f1 - f3)) / (2 * (f1 - 2 * f2 + f3))
    return [x1, x2, x3, xApprox]

def dscPowell(x0, grad, eps, lmb, delta):
    dsc_res = dsc(x0, grad, lmb, delta)
    a = dsc_res[0]
    xmin = dsc_res[1]
    b = dsc_res[2]
    xApprox = dsc_res[3]

    while abs(xmin-xApprox) >= eps or abs(function(calcX(x0, grad, xmin)) - function(calcX(x0, grad, xApprox))) >= eps:
        if xApprox < xmin:
            b = xmin
        else:
            a = xmin
        xmin = xApprox
        funcRes =  [function(calcX(x0, grad, a)), function(calcX(x0, grad, xmin)), function(calcX(x0, grad, b))]
        a1 = (funcRes[1] - funcRes[0]) / (xmin - a)
        a2 = ((funcRes[2] - funcRes[0]) / (b - a) - a1) / (b - xmin)
        xApprox = (a + xmin) / 2 - a1 / (2 * a2)
    return xmin

def Rpartan(x0, eps1,lmb, color):
    STOP = False 
    STOP_R = False 
    grad = gradient(x0)
    count_dots = 0
    while not STOP_R:
        STOP = False
        while not STOP:
            xs = []
            xs.append(x0)
            print ("x0 = " + str(x0))
            lmb = calcLambda(x0, grad, eps1, lmb, DELTA_MOP)
            x1 = calcX(x0, grad, lmb)
            print ("x1 = " + str(x1))
            count_dots = count_dots + 1
            grad = gradient(x1)
            lmb = calcLambda(x1, grad, eps1, lmb, DELTA_MOP)
            x2 = calcX(x1, grad, lmb)
            print ("x2 = " + str(x2))
            count_dots = count_dots + 1
            grad = difference(x2, x0)
            lmb = calcLambda(x2, grad, eps1, lmb, DELTA_MOP)
            x0 = calcX(x2, grad, lmb)
            print ("x3 = " + str(x0))
            count_dots = count_dots + 1
            grad = gradient(x0)
            
            print("-------------------------")
            xs.append(x0) 
            num = (count_func+count_dots)%len(color)
            plot(xs, color[num])
            if not NormKO(grad, eps1):
                STOP = True
        if not NormKO(grad, eps1):
            STOP_R = True
        else:
            global R_func
            R_func = (R_func / 10)
            print(f'R: {R_func}')
    return count_dots, x0


def NormKO(grad, eps):
    if(norm(grad) <= eps):
        return False
    return True

def partan_run():
    CountToZero()
    count_dots, x0 = Rpartan([-1.2,0.0],eps1=0.001, lmb =0.001, color = ['orange',  'yellow', 'red', 'purple'])
    
    circle = pl.Circle((-1, 0), 1, fill = False, color = 'b')
    pl.gca().add_patch(circle)
    pl.grid()
    pl.show()
    print("FUNCTIONS COUNT = " + str(count_func))
    print("STEPS COUNT = " + str(count_dots))
    print("MIN FUNCTION = " + str(function(x0)))

if __name__ == '__main__':
    partan_run()
    
    
    
    