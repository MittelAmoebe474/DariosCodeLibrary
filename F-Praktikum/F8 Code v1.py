#!/usr/bin/env python
# coding: utf-8

# # F8 Fortgeschrittenen Praktikum
# ## Magnetische Flussdichte
# Plot der Magnetischen Flussdichte mit linearer Regression und Export als .pgf zum Integrieren in Latex

# In[41]:


import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import scipy.integrate as integrate
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

A_x = list()
A_1 = list()
A_2 = list()
A_3 = list()
A_4 = list()

with open("E:/Documents/Dokumente/Uni/6. Semester/PPF/F8/F8_Magnetfeld_Aufgabe2_1.csv", "r") as data:
    for num,line in enumerate(data):
        if num > 0:
            rem = line.split(",")
            A_x.append(float(rem[0]))
            A_1.append(float(rem[1]))
            A_2.append(float(rem[2]))
            A_3.append(float(rem[3]))
            A_4.append(float(rem[4]))

B_1 = integrate.trapezoid(A_1,x=A_x) / 7
B_2 = integrate.trapezoid(A_2,x=A_x) / 7
B_3 = integrate.trapezoid(A_3,x=A_x) / 7
B_4 = integrate.trapezoid(A_4,x=A_x) / 7

print(B_1,B_2,B_3,B_4)

m, b = np.polyfit([0.8,1.0,1.2,1.6],[B_1,B_2,B_3,B_4], 1)
print(m,b)

plt.clf()
plt.plot([0.8,1.0,1.2,1.6],[B_1,B_2,B_3,B_4], 'ko', label='magnetische Flussdichte')
plt.plot([0.8,1.0,1.2,1.6],[m*x+b for x in [0.8,1.0,1.2,1.6]], 'r', label='lineare Regression mit polyfit')
plt.axis([0.5,2,6,13])
plt.xlabel('Strom in A')
plt.ylabel('Magnetflussdichte in mT')
plt.legend()

m_s, b_s, r_value, p_value, std_err = stats.linregress([0.8,1.0,1.2,1.6],[B_1,B_2,B_3,B_4])
print(m_s, b_s, r_value, p_value, std_err)

plt.savefig('plotneu.pgf')


# ## Bestimmung von Kontrast und Polarisationsgrad

# In[18]:


import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import math

def func_mallus(x,I_0,a):
    y = I_0*np.cos(x*math.pi/180-a)**2
    return y

x = np.linspace(0,360,num=37)
y = [7.925,7.178,6.114,4.677,3.34,1.87,0.829,0.169,0.01515,0.33,1.069,2.1043,3.385,4.627,5.944,6.973,7.644,7.929,7.738,7.068,6.007,4.722,3.252,1.936,0.859,0.219,0.01152,0.264,0.93,2.011,3.237,4.487,5.827,6.973,7.654,8.024,7.833]

p, c = curve_fit(func_mallus, x, y)
print(c)
print(p[0])
print(p[1])
SE = np.sqrt(np.diag(c))
print(SE)

plt.clf()
plt.plot(x,y,'ko', label='Intensität')
plt.plot(np.linspace(0,360,num=361),func_mallus(np.linspace(0,360,num=361),p[0],p[1]), 'r', label='Curve Fit')
plt.xlabel('Winkel des Polarisators in Grad')
plt.ylabel('gemessene Intensität')
plt.legend()
plt.savefig('plot2.pgf')


# In[19]:


import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(0,7,num=15)
y = [3.57,4.75,5.675,6.435,7.20,7.48,7.73,7.76,7.60,7.36,6.87,6.04,5.06,4.19,3.35]

plt.clf()
plt.plot(x,y,'ko', label='Magnetfeld')
plt.xlabel('Abstand vom linken Rand des Mediums in cm')
plt.ylabel('Magnetfeld in mT')
plt.legend()
plt.savefig('plot3.pgf')


# In[1]:


from jupyterthemes import get_themes
import jupyterthemes as jt
from jupyterthemes.stylefx import set_nb_theme
set_nb_theme('chesterish')


# In[ ]:




