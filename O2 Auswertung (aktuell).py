#!/usr/bin/env python
# coding: utf-8

# # O2 Nd:YAG Laser
# ## 1.1.1 Temperaturabhängigkeit 
# 
# Die Temperaturabhängigkeit der
# Emissionswellenlänge der Laserdiode ist für den maximalen Injektionsstrom und
# einen weiteren Injektionsstrom knapp oberhalb der Laserschwelle aufzunehmen.
# 

# In[5]:


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


plt.savefig('temperaturabhängigkeit.png')
plt.show()


# ## 1.1.2 Absorptionsspektrum 
# 
# Im Wellenlängen-(Temperatur-)Bereich der Laserdiode
# ist bei maximalem Diodenstrom das Absorptionsspektrum des Nd:YAG-Kristalls
# aufzunehmen und die für den späteren laserbetrieb optimale Pumpwellenlänge
# zu bestimmen.
# 

# In[106]:


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
absorption = np.log10(Pe/Pk)

# Plot the data
plt.clf()
plt.plot(wavelength,absorption, 'ro', label='Absorptionsspektrum',markersize=4)
plt.xlabel(r'Wellenlänge in nm')
plt.ylabel(r'Absorptionsspektrum in a.u.')
plt.savefig('absorptionsspektrum.png')
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

# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.3_Kennlinie.csv')

# Extract the data from the DataFrame 
I = data['mA']
Poptimal = data['32_5[mW]']
Plow = data['15_0[mW]']

# Linear fit to determine nullpunkt
res2 = stats.linregress(I[:-12], Poptimal[:-12])

slope2 = res2.slope
intercept2 = res2.intercept
print('Slope: ',slope2)
print('intercept: ',intercept2)

nullstelle2 = abs(intercept2/slope2)
print('nullstelle: ',nullstelle2)

# Determine Unsicherheiten

print('R- Wert', res2.rvalue)
print('slope stderr', res2.stderr)
print('intercept stderr', res2.intercept_stderr)

uncertainty2 = abs(res2.intercept_stderr/res2.slope) + abs(res2.intercept/(res2.slope**2)*res2.stderr)
print('uncertainty: ',uncertainty2)

# Plot the data
plt.clf()
plt.plot(I, Poptimal, 'ro', label=r'T$ = 32,5^\circ$C',markersize=4)
plt.plot(I, Plow ,'bo', label=r'T$ = 15^\circ$C',markersize=4)
plt.xlabel(r'Strom in mA')
plt.ylabel(r'Leistung in mW')
plt.legend(loc='upper left')
plt.savefig('leistungskennlinien.png')
plt.show()


# ## 1.1.4 Lebensdauermessung
# 
# Aus der Echtzeitmessung der spontanen Emission des
# Nd:YAG-Kristalls bei rechteckmoduliertem Pumpen ist die Lebensdauer des
# metastabilen Laserniveaus zu bestimmen.

# In[8]:


import pandas as pd
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
ax1.axvline(seconds[865638], color='orange', linestyle='--',label='systematischer Fehler')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.savefig('lebensdauer.png')
plt.show()


# # 1.2.2 Laserleistung
# Ist der Laser optimal justiert, d.h. die Ausgangsleistung beträgt
# mindestens 80 mW, wird die Laserleistung als Funktion der Pumpleistung des
# Diodenlasers im cw-Betrieb aufgenommen und die Laserschwelle sowie diverse
# Wirkungsgrade bestimmt (siehe unten).

# In[58]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats
from scipy import constants

# Read the CSV files
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.2.2_NdYAG_Leistung.csv')

# Extract the data from the DataFrame 
volt = data['mA']
power = data['P_NL[mW]']

# Additional parameters
transmission = (0.7085+0.732)/2 # transmission from the SCHOTT data sheets for 1060nm and 1070nm
power_adj = power / transmission

# Linear fit to determine nullpunkt
res = stats.linregress(volt[1:19+1], power_adj[1:19+1])

slope = res.slope
intercept = res.intercept
print('Slope: ',slope)
print('intercept: ',intercept)

nullstelle = abs(intercept/slope)
print('nullstelle: ',nullstelle)

# Determine Unsicherheiten

print('R- Wert', res.rvalue)
print('slope stderr', res.stderr)
print('intercept stderr', res.intercept_stderr)

uncertainty = abs(res.intercept_stderr/res.slope) + abs(res.intercept/(res.slope**2)*res.stderr)
print('uncertainty: ',uncertainty)

# Determine sigma_s 
P_th = res2.slope * nullstelle + res2.intercept
print('Isso :',P_th)
P_a = power_adj[0:19]
P_p = Poptimal[:-13]

res3 = stats.linregress(P_p - P_th, P_a)

slope3 = res3.slope
intercept3 = res3.intercept
print('Slope: ',slope3)
print('intercept: ',intercept3)

# Determine Unsicherheiten

print('R- Wert', res3.rvalue)
print('slope stderr', res3.stderr)

P_th_un = np.sqrt(abs(res2.slope*uncertainty)**2 + abs(nullstelle*res2.stderr)**2 + abs(res2.intercept_stderr)**2)
print('Pth Unsicherheit: ',P_th_un )

# Determine optischer Wirkungsgrad

mean = np.mean([power_adj[0:19]/Poptimal[:-13]])
mean_un = np.std([power_adj[0:19]/Poptimal[:-13]])
print('meanopt: ',mean)
print('meanopt unsich: ',mean_un)

# Determine gesamt Wirkungsgrad

mean2 = np.mean([power_adj[0:19]/(volt[0:19]*1.9)])
mean2_un = np.std([power_adj[0:19]/(volt[0:19]*1.9)])
print('meanges: ',mean2)
print('meanopt unsich: ',mean2_un)

# Determine Pumpwirkungsgrad
E32 = constants.c / 1064e-9 * constants.h
E41 = constants.c / 808.3e-9 * constants.h
nu = res3.slope * E41 / E32
nu_un = res3.stderr * E41 / E32
print('nu: ',nu)
print('nuun: ',nu_un)

# Plot the data
plt.clf()
plt.plot(volt, power_adj, 'ro', label=r'T = $32.5^\circ$C',markersize=4)
plt.plot(volt,slope*volt+intercept, 'r')
plt.xlabel(r'Injektionsstrom in mA')
plt.ylabel(r'Laserleistung in mW')
plt.legend(loc='upper left')

plt.savefig('laserkennlinie.png')
plt.show()


# # 1.2.3 Wellenlängenabhängigkeit
# Die aufzunehmende Abhängigkeit der
# Laserleistung von der Pumpwellenlänge ist mit dem Absorptionsspektrum des
# Nd:YAG-Kristalls zu vergleichen.

# In[129]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Read the CSV file
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.1GaAs_T_lambda.csv')
data2 = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.2.3_PNd_T.csv')
data3 = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.2_NdYAG_Abs.csv')

# Extract

wavelength1 = data['620mA_nm']
absorption = np.log10(data3['P_e[mW]'] / data3['P_k'])
random = data3['P_e[mW]'] - data3['P_k']
P_L = data2['P_Nd[mW]']

# Plot
plt.clf()
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
line1, = ax1.plot(wavelength1, P_L[::-1], 'bo', label='Leistung Nd:YAG')

ax1_twin = ax1.twinx()
ax1.set_ylabel('Leistung in mW')
line2, = ax1_twin.plot(wavelength1, absorption, 'ro', label='Absorptionsspektrum')
ax1_twin.set_ylabel('Absorptionsspektrum')

line3, = ax2.plot(wavelength1, P_L[::-1], 'bo', label='Leistung Nd:YAG')
ax2.set_ylabel('Leistung in mW')

ax2_twin = ax2.twinx()
line4, = ax2_twin.plot(wavelength1, random, 'go', label='absorbierte Leistung')
ax2_twin.set_ylabel('Leistung mW')
ax2.set_xlabel('Wellenlänge in nm')


# Combine the handles and labels from both subplots into a single legend
lines = [line1, line2, line4]
labels = [line.get_label() for line in lines]
ax2.legend(lines, labels, loc='lower right')

plt.subplots_adjust(hspace=0.4)
plt.savefig('warme.png')
plt.show()


# # 1.2.4 Spiking
# Mit Hilfe des modulierten Pumpens sind die Nichtgleichgewichts-
# Prozesse beim Einschalten des Lasers (spiking) bei zwei verschiedenen
# Pumpströmen zu registrieren.

# In[146]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\dario\spi_400.csv')
data2 = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\dario\spi_620.csv')

# Extract
second1 = data['Second']*1000
second2 = data2['Second']*1000

volt1 = data['Volt']
volt2 = data2['Volt']

# PLot
plt.clf()

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 6))

ax1.plot(second1 ,volt1, 'r',label=r'I = 400mA',markersize=0.3)
ax1.set_xlim(0,0.175)
ax2.set_xlabel('Sekunden in ms')
ax1.set_ylabel('Spannung in V')
ax2.set_ylabel('Spannung in V')
ax2.plot(second2 ,volt2, 'b',label=r'I = 620mA',markersize=0.3)
ax2.set_xlim(0,0.175)
ax1.legend(loc='upper right')
ax2.legend(loc='upper right')
plt.savefig('spike.png')
plt.show()


# # 1.2.5 Güteschaltung
# Nach Einbau des sättigbaren Absorbers (Cr4+:YAG) soll die
# zeitliche Struktur der Pulse mit einer Photodiode aufgenommen werden. Es sind
# die Pulsdauer, die Wiederholrate und die Spitzenleistung der Pulse zu bestimmen.
# Damit die Güteschaltung funktioniert, muss der Laser optimal justiert sein, d.h. die
# Justage ist vor diesem Versuchsabschnitt nochmals zu überprüfen und ggf. zu
# optimieren.

# In[488]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
import math
from scipy.optimize import newton

# Read the CSV file
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\dario\125_puls.csv')

# Extract
seconds = data['Second']
volt = data['Volt']

# Find Peaks
peaks, _ = find_peaks(volt)
peaks_new = [value for value in peaks if volt[value] > 0.08]
peak1 = [value for value in peaks_new if (seconds[value] >= -0.4/1000)&(seconds[value] <=-0.3/1000)]
peak6 = [value for value in peaks_new if (seconds[value] >= 0.4/1000)&(seconds[value] <=0.6/1000)]
print('rate: ',6/abs(seconds[min(peak6)]-seconds[min(peak1)]))

# Define the Gaussian function
def gaussian(x, amplitude, mean, std_dev):
    return amplitude * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))

# Gaussian Range
secmin = (seconds - (-0.3515/1000)).abs().idxmin()
secmax = (seconds - (-0.35075/1000)).abs().idxmin()
print(secmin, secmax)

# Perform curve fitting
weights = np.ones_like(volt)[secmin:secmax]
print(volt.index[min(peak1):max(peak1)+1]-volt.index[secmin])
weights[373:445] = 0.0001
weights[:444] = 100
weights[446:] = 1
print(len(weights))
params, _ = curve_fit(gaussian, seconds[secmin:secmax]*1000, volt[secmin:secmax], p0=[0.8, -0.3511725, 0.0000285], sigma = weights)  # p0 provides initial parameter estimates
print(params)

# Evaluate the Gaussian function with the fitted parameters
y_smooth = gaussian(seconds[min(peak1):secmax]*1000, *params)

# Plot
plt.clf()

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 6))
ax1.plot(seconds*1000, volt, 'ro', label='Güteschaltung', markersize=0.3)
ax1.plot(seconds[peaks_new]*1000, volt[peaks_new], 'bo', markersize=0.3)
#ax1.plot(seconds[secmin:secmax]*1000,gaussian(seconds[secmin:secmax]*1000,0.16,-0.351175,0.00003),color='red')
ax1.plot(seconds[secmin:max(peak1)]*1000,gaussian(seconds[secmin:max(peak1)]*1000,0.18,-0.3511725,0.0000285),color='orange',label='Gauss Anpassung')
ax1.plot(seconds[min(peak1):secmax]*1000,y_smooth,color='blue',label='exponentieller Abfall')

ax1.set_xlim(-0.3515,-0.35075)
ax1.set_xlabel(r'Sekunden in ms')
ax1.set_ylabel('Spannug in V')
ax1.legend(loc='upper right')

ax1.axvline(min(seconds[peak1])*1000, color='orange', linestyle='--')
ax1.axvline(max(seconds[peak1])*1000, color='orange', linestyle='--')
ax1.axvline(-0.351183,color='green')
ax1.axhline(0.168)
ax1.axhline(0.168/2)

ax2.plot(seconds*1000, volt, 'ro', label='Güteschaltung', markersize=0.3)
ax2.plot(seconds[peaks_new]*1000, volt[peaks_new], 'bo', label='Peaks', markersize=0.3)
ax2.legend(loc='upper right')
ax2.set_xlabel(r'Sekunden in ms')
ax2.set_ylabel('Spannug in V')
plt.savefig('guete.png')
plt.show()


# # 1.2.7 Frequenzverdopplung
# Nach Einbau eines Verdopplerkristalls in den Resonator
# und unter Verwendung eines höher reflektierenden Auskoppelspiegels (Warum?)
# ist die grüne Ausgangsleistung als Funktion der Pumpleistung zu messen.
# 

# In[497]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.2.7_Frequ_x2.csv')
data2 = pd.read_csv(r'E:\Documents\Dokumente\Uni\6. Semester\PPF\O2\1.1.3_Kennlinie.csv')

# Extract 
P_pump = 0.9432556390977445*data['mA']-216.1799248120301
P = data['P[mricroW]']

# plot
plt.clf()
plt.plot(P_pump,P,'ro')
plt.ylabel('Ausgangsleistung in mW')
plt.xlabel('Pumpleistung in mW')
plt.savefig('freqx2.png')
plt.show()

