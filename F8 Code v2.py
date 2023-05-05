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


# ## Bestimmung der Verdet Konstante

# In[60]:


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
import scipy.stats as stats
import math

purple = [0.8,1,1.2,1.56]
purple_DC = 2.5
purple_PP = [0.0632,0.078,0.0924,0.120]
green = [0.83,1,1.2,1.4]
green_DC = 6.21
green_PP = [0.1141,0.136,0.1633,0.190]
orange = [0.8,1,1.2,1.4]
orange_DC = 4.26
orange_PP = [0.0541,0.0675,0.0802,0.0937]
red = [0.8,1,1.2,1.4]
red_DC = 3
red_PP = [0.0378,0.0477,0.0561,0.0658]
red_laser = [0.6,0.8,1,1.2]
red_laser_DC = 1.409
red_laser_PP = [0.0536,0.0713,0.0884,0.1049]

def drehwinkel(I_DC, I_PP):
    theta = I_PP/(4*I_DC)
    return theta
def magnetfeld(I):
    B = 8.067*I-0.1158
    return B

plt.clf()
plt.plot([math.sqrt(2)*magnetfeld(x) for x in purple], [drehwinkel(purple_DC,y) for y in purple_PP], 'mo', label='Purple LED')
plt.plot([math.sqrt(2)*magnetfeld(x) for x in green], [drehwinkel(green_DC,y) for y in green_PP], 'go', label='Green LED')
plt.plot([math.sqrt(2)*magnetfeld(x) for x in orange], [drehwinkel(orange_DC,y) for y in orange_PP], 'yo', label='Orange LED')
plt.plot([math.sqrt(2)*magnetfeld(x) for x in red], [drehwinkel(red_DC,y) for y in red_PP], 'ro', label='Red LED')

m_1, b_1 = np.polyfit([math.sqrt(2)*magnetfeld(x) for x in purple], [drehwinkel(purple_DC,y) for y in purple_PP],1)
m_2, b_2 = np.polyfit([math.sqrt(2)*magnetfeld(x) for x in green],[drehwinkel(green_DC,y) for y in green_PP],1)
m_3, b_3 = np.polyfit([math.sqrt(2)*magnetfeld(x) for x in orange], [drehwinkel(orange_DC,y) for y in orange_PP],1)
m_4, b_4 = np.polyfit([math.sqrt(2)*magnetfeld(x) for x in red],[drehwinkel(red_DC,y) for y in red_PP],1)

plt.plot([math.sqrt(2)*magnetfeld(x) for x in purple],[m_1*x+b_1 for x in[math.sqrt(2)*magnetfeld(x) for x in purple]], 'm')
plt.plot([math.sqrt(2)*magnetfeld(x) for x in green],[m_2*x+b_2 for x in[math.sqrt(2)*magnetfeld(x) for x in green]],'g')
plt.plot([math.sqrt(2)*magnetfeld(x) for x in orange],[m_3*x+b_3 for x in[math.sqrt(2)*magnetfeld(x) for x in orange]],'y')
plt.plot([math.sqrt(2)*magnetfeld(x) for x in red],[m_4*x+b_4 for x in[math.sqrt(2)*magnetfeld(x) for x in red]],'r')

m_s, b_s, r_value, p_value, std_err = stats.linregress([math.sqrt(2)*magnetfeld(x) for x in purple], [drehwinkel(purple_DC,y) for y in purple_PP])
print(m_s, b_s, r_value, p_value, std_err)
m_s, b_s, r_value, p_value, std_err = stats.linregress([math.sqrt(2)*magnetfeld(x) for x in green],[drehwinkel(green_DC,y) for y in green_PP])
print(m_s, b_s, r_value, p_value, std_err)
m_s, b_s, r_value, p_value, std_err = stats.linregress([math.sqrt(2)*magnetfeld(x) for x in orange], [drehwinkel(orange_DC,y) for y in orange_PP])
print(m_s, b_s, r_value, p_value, std_err)
m_s, b_s, r_value, p_value, std_err = stats.linregress([math.sqrt(2)*magnetfeld(x) for x in red],[drehwinkel(red_DC,y) for y in red_PP])
print(m_s, b_s, r_value, p_value, std_err)

plt.xlabel('Amplitude der magnetischen Flussdichte in mT')
plt.ylabel('Drehwinkel in rad')
plt.legend()
plt.savefig('plot4.pgf')


# ## Abbildung der Wellenlängenabhängigkeit der Verdet Konstante

# In[36]:


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

y = [163.19,117.46,84.30,84.12]
x = [398,517,607,629]

xerr_purple = 13/2
xerr_green = 37/2
xerr_orange = 17/2
xerr_red = 16/2

plt.clf()
plt.plot(x,y,'ko')
plt.errorbar(x[0],y[0],xerr=xerr_purple, fmt='')
plt.errorbar(x[1],y[1],xerr=xerr_green, fmt='')
plt.errorbar(x[2],y[2],xerr=xerr_orange, fmt='')
plt.errorbar(x[3],y[3],xerr=xerr_red, fmt='')
plt.xlabel('Wellenlänge in nm')
plt.ylabel('Verdetsche Konstante in rad T-1 m-1')

plt.savefig('plot5.pgf')


# ## Sellmeier Gleichung für BK7 und SF10 Glas

# In[58]:


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
import scipy.stats as stats

wavelengths = [398,517,607,629]
n_BK7 = [1.5311,1.5203,1.5160,1.5152]

m,b = np.polyfit([1/x**2 for x in wavelengths],[1/(y**2-1) for y in n_BK7],1)
print(m,b)
m_s, b_s, r_value, p_value, std_err = stats.linregress([1/x**2 for x in wavelengths],[1/(y**2-1) for y in n_BK7])
print(m_s, b_s, r_value, p_value, std_err)

plt.clf()
plt.plot([1/x**2 for x in wavelengths],[1/(y**2-1) for y in n_BK7],'ro')
plt.plot([1/x**2 for x in wavelengths],[m*x+b for x in [1/x**2 for x in wavelengths]], 'r')
plt.savefig('plot6.pgf')


# In[3]:


from jupyterthemes import get_themes
import jupyterthemes as jt
from jupyterthemes.stylefx import set_nb_theme
set_nb_theme('chesterish')


# In[ ]:




