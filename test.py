import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

ts = []
n = 1900
s = 10
for t in range(0,10000):
    a = math.exp(-(t*s**2/n))
    b = math.cos(math.pi**2/s*t)*n/4+n*3/4
    y = (a*b)//1
    ts.append(y)
    if y==1:
        break
print(min(ts),max(ts))
plt.plot(ts)
plt.show()
