#!/usr/bin/env python
# coding: utf-8

# # O2 Nd:YAG Laser
# ## 1.1.1 Temperaturabhängigkeit 
# 
# Die Temperaturabhängigkeit der
# Emissionswellenlänge der Laserdiode ist für den maximalen Injektionsstrom und
# einen weiteren Injektionsstrom knapp oberhalb der Laserschwelle aufzunehmen.
# 

# In[31]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.1GaAs_T_lambda.csv')

# Extract data from DataFrame
temp = data['temp']
wavelength1 = data['620mA_nm']
wavelength2 = data['210mA_nm']

# Plot the data
plt.clf()
plt.plot(temp, wavelength1, 'ro', label='I = 620mA',markersize=4)
plt.plot(temp,wavelength2,'bo', label='I = 210mA',markersize=4)
plt.xlabel(r'Temperatur in $^\circ$C')
plt.ylabel(r'Emissionswellenlänge in nm')
plt.legend(loc='upper left')
plt.show()


# ## 1.1.2 Absorptionsspektrum 
# 
# Im Wellenlängen-(Temperatur-)Bereich der Laserdiode
# ist bei maximalem Diodenstrom das Absorptionsspektrum des Nd:YAG-Kristalls
# aufzunehmen und die für den späteren laserbetrieb optimale Pumpwellenlänge
# zu bestimmen.
# 

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.2_NdYAG_Abs.csv')
data2 = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.1GaAs_T_lambda.csv')

# Extract data from DataFrames
temp = data['temp']
wavelength = data2['620mA_nm']
Pe = data['P_e[mW]']
Pk = data['P_k']

# Calculate the relative absorption
absorption = Pe / Pk

# Plot the data
plt.clf()
plt.plot(wavelength,absorption, 'ro', label='Absorptionsspektrum',markersize=4)
plt.xlabel(r'Wellenlänge in nm')
plt.ylabel(r'relative Absorption')
plt.show()


# ## 1.1.3 Leistungskennlinien
# 
# Die Abhängigkeit der Austrittsstrahlungsleistung der
# Laserdiode vom Injektionsstrom ist für zwei verschiedene Temperaturen zu
# ermitteln (ohne Nd:YAG-Kristall!): 
# 
# (i) bei der zuvor bestimmten optimalen
# Temperatur maximaler Nd:YAG-Absorption (wichtige Kalibrierung für spätere
# Messungen bzw. deren Auswertung) 
# 
# (ii) bei einer anderen selbstgewählten
# Temperatur. Wählen Sie ein hinreichend großes Messintervall, welches auch die
# Laserschwelle beinhaltet!

# In[36]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.3_Kennlinie.csv')

# Extract the data from the DataFrame 
I = data['mA']
Poptimal = data['32_5[mW]']
Plow = data['15_0[mW]']

# Plot the data
plt.clf()
plt.plot(I, Poptimal, 'ro', label=r'T$ = 32,5^\circ$C',markersize=4)
plt.plot(I, Plow ,'bo', label=r'T$ = 15^\circ$C',markersize=4)
plt.xlabel(r'Strom in mA')
plt.ylabel(r'Leistung in mW')
plt.legend(loc='upper left')
plt.show()


# ## 1.1.4 Lebensdauermessung
# 
# Aus der Echtzeitmessung der spontanen Emission des
# Nd:YAG-Kristalls bei rechteckmoduliertem Pumpen ist die Lebensdauer des
# metastabilen Laserniveaus zu bestimmen.

# In[142]:


# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


# Read the CSV files
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.4c1.csv')
data2 = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.4c3.csv')

# Extract the data from the DataFrame 
seconds = data['Second']
volt = data['Volt']

seconds2 = data2['Second']
volt2 = data2['Volt']

# Find the indices of value within the range
indices = volt2.index[(volt2 >= 0.155) & (volt2 <= 0.16)].tolist()
indices2 = volt2.index[(volt2 >= 0.11) & (volt2 <= 0.1225)].tolist()

indices3 = volt.index[(volt >= 0.6) & (volt <= 0.8)].tolist()

# Calculate the mean
mean_value = np.mean(volt2[indices])
mean_value2 = np.mean(volt2[indices2])

print('Halbwertspannung',(mean_value - mean_value2)/math.e + mean_value2)

# Find Halbwertszeit
indices4 = seconds.index[(seconds >= -0.002) & (seconds <= -0.001)].tolist()
diffvolt = 1
for i in indices4:
        if diffvolt >= abs((mean_value - mean_value2)/math.e + mean_value2 - volt2[i]):
            diffvolt = abs((mean_value - mean_value2)/math.e + mean_value2 - volt2[i])
            indice = i
print('Unsicherheit',diffvolt)
print('Index',indice)
print('Halbwertszeit1 ',abs(seconds[indice]-seconds[408799]))

indices5 = seconds.index[(seconds >= 0.001) & (seconds <= 0.003)].tolist()
diffvolt = 1
for i in indices5:
        if diffvolt >= abs((mean_value - mean_value2)/math.e + mean_value2 - volt2[i]):
            diffvolt = abs((mean_value - mean_value2)/math.e + mean_value2 - volt2[i])
            indice = i
print('Unsicherheit',diffvolt)
print('Index',indice)
print('Halbwertszeit2 ',abs(seconds[indice]-seconds[408799]))

# Systematische Unsicherheit korrigieren bei 2. Halbwertszeit
indices6 = seconds.index[(seconds >= seconds[850711]) & (seconds <= seconds[850711]+0.001) & (volt2 >= mean_value-0.001) & (volt2 <= mean_value+0.001)].tolist()
diffsec = 0
for i in indices6:
        if diffsec <= abs(seconds[850711] - seconds[i]):
            diffsec = abs(seconds[850711] - seconds[i])
            indice2 = i
print('Unsicherheit',diffsec)
print('Index',indice2)
print('Systematische Unsicherheit HWZ 2 ',abs(seconds[indice2]-seconds[408799]))

# korrigierte HWZ 2
print('HWZ2 korrigiert', abs(seconds[indice]-seconds[408799]) - abs(seconds[indice2]-seconds[408799]))

# Plot the data
plt.clf()

fig, ax1 = plt.subplots()
ax1.plot(seconds ,volt, 'ro',label='Rechteckspannung',markersize=0.5)
plt.xlabel(r'Sekunden in s')
ax1.set_ylabel(r'Rechteckspannung in V')
ax1.set_ylim(-0.2,1.6)

ax2 = ax1.twinx()
ax2.plot(seconds2 ,volt2, 'bo',label='Photodiodenspannung',markersize=0.5)
ax2.set_ylabel(r'Photodiodenspannung in V')
ax2.set_ylim(0.11,0.18)
ax2.axhline(mean_value, color='green', linestyle='--')
ax2.axhline(mean_value2, color='green', linestyle='--')
ax2.axhline((mean_value - mean_value2)/math.e + mean_value2, color='yellow', linestyle='--', label='Halbwertspannung')
ax1.axvline(seconds[408799], color='green', linestyle='--')
ax1.axvline(seconds[850711], color='green', linestyle='--')
# unsicherheit
ax1.axvline(seconds[865638], color='orange', linestyle='--')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()


# In[ ]:




