import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import numpy as np

def integrand(t,T):
    return t*(9.0/T+T/4.0-0.5*t)+(9.0/T+T/4.0-0.5*t)**2

N=1000
MAX_T=20
T_array=np.linspace(0.3,MAX_T,N)
J_array=np.zeros(N)

for i in range (N):
    J_array[i] = integrate.quad(integrand, 0, T_array[i], args=(T_array[i]))[0]

plt.title("J(T) dependence on the free time T")
plt.xlabel("free time T [seconds]")
plt.ylabel("J(t) (adimensional)")
plt.plot(T_array,J_array, 'r')
plt.show()

