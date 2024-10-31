import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.visualization import quantity_support
quantity_support()  # for getting units on the axes below

c = 299792458

# Now I want to recreate some plots from the LATEST Guo data.
# I have access to table 4, so I will recreate figure 1:
# Specifically I will recrate the histograms from the bottom panels of figure 1.
# For the middle panel, I will use Topcat to recreate the grey plot (I don't have access to which group each CLAGN is in).
# I won't recreate the top panel as I don't have access to the parent sample data.

table_4_GUO = pd.read_csv('guo23_table4_clagn.csv')

# Step 2: Filter the data based on the 'transition Line' column
turn_on_z = table_4_GUO[table_4_GUO['transition'] == 'turn-on']['Redshift']
turn_off_z = table_4_GUO[table_4_GUO['transition'] == 'turn-off']['Redshift']
# Includes CLAGN with more than 1 BEL

# Step 3: Create the histogram
plt.figure(figsize=(10, 6))

my_list_x_axis_z = np.arange(0, 3.0, 0.2).tolist()

# Plot histogram for 'turn-on'
plt.hist(turn_on_z, bins=my_list_x_axis_z, histtype='step', label='Turn-on', color='blue')

# Plot histogram for 'turn-off'
plt.hist(turn_off_z, bins=my_list_x_axis_z, histtype='step', label='Turn-off', color='red')

# Adding labels and title
plt.xticks(my_list_x_axis_z)
plt.xlim(0)  # Sets the minimum x-axis value to 0
plt.xlabel('Redshift')
plt.ylabel('N')
plt.title('CLAGN Redshift: Turn-on vs Turn-off')
plt.legend()

# Show the plot
plt.show()

# Same plot but for r_band magnitude
turn_on_r_mag = table_4_GUO[table_4_GUO['transition'] == 'turn-on']['r(mag)']
turn_off_r_mag = table_4_GUO[table_4_GUO['transition'] == 'turn-off']['r(mag)']

# Step 3: Create the histogram
plt.figure(figsize=(20, 6))

my_list_x_axis_r_mag = np.arange(18, 22.6, 0.2).tolist()

# Plot histogram for 'turn-on'
plt.hist(turn_on_r_mag, bins=my_list_x_axis_r_mag, histtype='step', label='Turn-on', color='blue')

# Plot histogram for 'turn-off'
plt.hist(turn_off_r_mag, bins=my_list_x_axis_r_mag, histtype='step', label='Turn-off', color='red')

# Adding labels and title
plt.xticks(my_list_x_axis_r_mag)
plt.xlim(18)
plt.xlabel('Magnitude (r-band)')
plt.ylabel('N')
plt.title('CLAGN r-band Magnitude: Turn-on vs Turn-off')
plt.legend()

# Show the plot
plt.show()

table_4_GUO['MJD_1'] = table_4_GUO['MJD_1'].fillna(method='ffill')
table_4_GUO['MJD_2'] = table_4_GUO['MJD_2'].fillna(method='ffill')
table_4_GUO['Redshift'] = table_4_GUO['Redshift'].fillna(method='ffill')

table_4_GUO['Velocity Parallel'] = c*(((1+table_4_GUO['Redshift'])**2)-1)/(((1+table_4_GUO['Redshift'])**2)+1)

table_4_GUO['Obs_MJD_diff'] = table_4_GUO['MJD_2'] - table_4_GUO['MJD_1']
table_4_GUO['Rest_MJD_diff'] = (table_4_GUO['MJD_2'] - table_4_GUO['MJD_1'])*(np.sqrt(1-(((table_4_GUO['Velocity Parallel'])**2)/(c**2))))

Obs_Time_scale_Halpha = table_4_GUO[table_4_GUO['Line'] == 'Halpha']['Obs_MJD_diff']
Obs_Time_scale_Hbeta = table_4_GUO[table_4_GUO['Line'] == 'Hbeta']['Obs_MJD_diff']
Obs_Time_scale_H = np.append(Obs_Time_scale_Halpha, Obs_Time_scale_Hbeta)
Obs_Time_scale_C3 = table_4_GUO[table_4_GUO['Line'] == 'C iii]']['Obs_MJD_diff']
Obs_Time_scale_C4 = table_4_GUO[table_4_GUO['Line'] == 'C iv']['Obs_MJD_diff']
Obs_Time_scale_C = np.append(Obs_Time_scale_C3, Obs_Time_scale_C4)
Obs_Time_scale_Mg2 = table_4_GUO[table_4_GUO['Line'] == 'Mg ii']['Obs_MJD_diff']
Obs_Time_scale_Mg2.tolist()

#Medians
median_Obs_Time_scale_H = np.median(sorted(Obs_Time_scale_H))
median_Obs_Time_scale_C = np.median(sorted(Obs_Time_scale_C))
median_Obs_Time_scale_Mg2 = np.median(sorted(Obs_Time_scale_Mg2))

plt.figure(figsize=(20, 6))

my_list_x_axis_Obs_Timescale = np.arange(0, 8000, 500).tolist()

plt.hist(Obs_Time_scale_C, bins=my_list_x_axis_Obs_Timescale, histtype='step', label='C III] + C IV', color='goldenrod')
plt.axvline(median_Obs_Time_scale_C, linestyle='--', linewidth=2, color='goldenrod')
plt.hist(Obs_Time_scale_Mg2, bins=my_list_x_axis_Obs_Timescale, histtype='step', label='MG II', color='green')
plt.axvline(median_Obs_Time_scale_Mg2, linestyle='--', linewidth=2, color='green')
plt.hist(Obs_Time_scale_H, bins=my_list_x_axis_Obs_Timescale, histtype='step', label='Halpha + Hbeta', color='blue')
plt.axvline(median_Obs_Time_scale_H, linestyle='--', linewidth=2, color='blue')

# Adding labels and title
plt.xticks(my_list_x_axis_Obs_Timescale)
plt.xlim(0)
plt.ylim(0, 20)
plt.xlabel('Timescale (days)')
plt.ylabel('N')
plt.title('CLAGN Observed Frame Timescale')
plt.legend()

plt.show()

Rest_Time_scale_Halpha = table_4_GUO[table_4_GUO['Line'] == 'Halpha']['Rest_MJD_diff']
Rest_Time_scale_Hbeta = table_4_GUO[table_4_GUO['Line'] == 'Hbeta']['Rest_MJD_diff']
Rest_Time_scale_H = np.append(Rest_Time_scale_Halpha, Rest_Time_scale_Hbeta)
Rest_Time_scale_C3 = table_4_GUO[table_4_GUO['Line'] == 'C iii]']['Rest_MJD_diff']
Rest_Time_scale_C4 = table_4_GUO[table_4_GUO['Line'] == 'C iv']['Rest_MJD_diff']
Rest_Time_scale_C = np.append(Rest_Time_scale_C3, Rest_Time_scale_C4)
Rest_Time_scale_Mg2 = table_4_GUO[table_4_GUO['Line'] == 'Mg ii']['Rest_MJD_diff']
Rest_Time_scale_Mg2.tolist()

median_Rest_Time_scale_H = np.median(sorted(Rest_Time_scale_H))
median_Rest_Time_scale_C = np.median(sorted(Rest_Time_scale_C))
median_Rest_Time_scale_Mg2 = np.median(sorted(Rest_Time_scale_Mg2))

plt.figure(figsize=(20, 6))

my_list_x_axis_Rest_Timescale = np.arange(0, 7000, 250).tolist()

plt.hist(Rest_Time_scale_C, bins=my_list_x_axis_Rest_Timescale, histtype='step', label='C III] + C IV', color='goldenrod')
plt.axvline(median_Rest_Time_scale_C, linestyle='--', linewidth=2, color='goldenrod')
plt.hist(Rest_Time_scale_Mg2, bins=my_list_x_axis_Rest_Timescale, histtype='step', label='MG II', color='green')
plt.axvline(median_Rest_Time_scale_Mg2, linestyle='--', linewidth=2, color='green')
plt.hist(Rest_Time_scale_H, bins=my_list_x_axis_Rest_Timescale, histtype='step', label='Halpha + Hbeta', color='blue')
plt.axvline(median_Rest_Time_scale_H, linestyle='--', linewidth=2, color='blue')

# Adding labels and title
plt.xticks(my_list_x_axis_Rest_Timescale)
plt.xlim(0)
plt.ylim(0, 20)
plt.xlabel('Timescale (days)')
plt.ylabel('N')
plt.title('CLAGN Rest Frame Timescale')
plt.legend()

plt.show()