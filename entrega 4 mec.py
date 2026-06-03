import numpy as np
import scipy.stats as ss
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#definicio de parametres
g=9.8
w=5
th_i=np.pi/4 #theta
d_th_i=0 #theta punt (derivem pi/4)
y0_1=[th_i,d_th_i]

t_rang=(0,40)
t=np.linspace(0,40,10000)

def edo_lin(t,y): #definim edo que depen del temps i (theta, d_theta)
    th1 = y[0] 
    d_th1 = y[1]
    r = np.exp(2*t) #r1
    dr = 2*r #la derivada de r1

    dd_th1=(-(2*(dr/r)*d_th1) + (((w)**2)*np.sin(th1)*np.cos(th1)) + ((g/r)*np.sin(th1))) #hem aillat $\ddot theta$ a l'edo
    
    return [d_th1, dd_th1]

def edo_var(t,y): #lo mismo pero canviarà r i R
    th2 = y[0] 
    d_th2 = y[1]
    r = 1+0.5*np.sin(2*t) #radi variable
    dr = np.cos(2*t) #la derivada de r

    dd_th2=(-(2*(dr/r)*d_th2) + (((w)**2)*np.sin(th2)*np.cos(th2)) + ((g/r)*np.sin(th2)))
    
    return [d_th2, dd_th2]

sol_lin = solve_ivp(edo_lin, t_rang, y0_1, t_eval=t,method="DOP853",rtol=1e-8,atol=1e-10) #mètode RK explicit, ordre 8
sol_var = solve_ivp(edo_var, t_rang, y0_1, t_eval=t,method="DOP853",rtol=1e-13,atol=1e-21) #rtol i atol toleràncies més petites del default (sistema molt caòtic)

plt.figure() #primera sol graficada
plt.plot(sol_lin.t, sol_lin.y[0], color='hotpink', label=r"$R(t)=e^{2t}$") #plot 1
plt.title('Solució per un radi lineal')
plt.legend()
plt.xlabel('t')
plt.ylabel(r'$\theta (t)$')
plt.grid()

plt.figure() #segona
plt.plot(sol_var.t, sol_var.y[0], color='hotpink', label=r"$R(t)=1+0.5\sin(2t)$")
plt.title('Solució per un radi variable')
plt.legend()
plt.xlabel('t')
plt.ylabel(r'$\theta (t)$')
plt.grid()

plt.figure() #la primera i la segona superposades, per comparar
plt.plot(sol_lin.t, sol_lin.y[0], color='hotpink', label=r"$R(t)=e^{2t}$") #plot 1
plt.plot(sol_var.t, sol_var.y[0], color='purple', label=r"$R(t)=1+0.5\sin(2t)$")#plot 2
plt.title("Solucions d'ambdós casos")
plt.legend()
plt.xlabel('t')
plt.ylabel(r'$\theta (t)$')
plt.grid()


plt.tight_layout
plt.show()