import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.convolution import convolve, Gaussian1DKernel
import pyspeckit
from specutils import Spectrum1D
from astropy.visualization import quantity_support
quantity_support()  # for getting units on the axes below

c = 299792458

# xaxis = np.linspace(-50,150,100)
# sigma = 10.
# center = 50.
# synth_data = np.exp(-(xaxis-center)**2/(sigma**2 * 2.))

# # Add noise
# stddev = 0.1
# noise = np.random.randn(xaxis.size)*stddev
# error = stddev*np.ones_like(synth_data)
# data = noise+synth_data

# # this will give a "blank header" warning, which is fine
# sp = pyspeckit.Spectrum(data=data, error=error, xarr=xaxis,
#                         xarrkwargs={'unit':'km/s'},
#                         unit='erg/s/cm^2/AA')

# sp.plotter()

# # Fit with automatic guesses
# sp.specfit(fittype='gaussian')

# # Fit with input guesses
# # The guesses initialize the fitter
# # This approach uses the 0th, 1st, and 2nd moments
# amplitude_guess = data.max()
# center_guess = (data*xaxis).sum()/data.sum()
# width_guess = (data.sum() / amplitude_guess / (2*np.pi))**0.5
# guesses = [amplitude_guess, center_guess, width_guess]
# sp.specfit(fittype='gaussian', guesses=guesses)

# sp.plotter(errstyle='fill')
# sp.specfit.plot_fit()


#Open the SDSS file
with fits.open("spec-8521-58175-0279.fits") as hdul:
    subset = hdul[1]        

    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
    units_sdss_flux = subset.data['flux'] * 10**-17 * u.Unit('erg cm-2 s-1 AA-1') #for fitting
    sdss_measured_wl = 10**subset.data['loglam'] #Wavelength in Angstroms
    units_sdss_measured_wl = 10**subset.data['loglam'] * u.AA 
    sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
     
#Open the DESI file
DESI_spec = pd.read_csv('spectrum_desi_152517.57+401357.6.csv')
wavelength_desi = DESI_spec.iloc[1:, 0]  # First column, skipping the first row (header)
flux_desi = DESI_spec.iloc[1:, 1]  # Second column, skipping the first row (header)

#Calculate rolling average manually
def rolling_average(arr, window_size):
    
    averages = []
    
    for i in range(len(arr) - window_size + 1):
        avg = np.mean(arr[i:i + window_size])
        averages.append(avg)
    
    # Return as a NumPy array
    return np.array(averages)

#Manual Rolling averages
SDSS_rolling = rolling_average(sdss_flux, 10)
DESI_rolling = rolling_average(flux_desi, 10)
sdss_measured_wl = sdss_measured_wl[9:]
wavelength_desi = wavelength_desi[9:]
sdss_flux = sdss_flux[9:]
flux_desi = flux_desi[9:]

# adjust stddev to control the degree of smoothing. Higher stddev means smoother
gaussian_kernel = Gaussian1DKernel(stddev=3)

# Smooth the flux data using the Gaussian kernel
Gaus_smoothed_SDSS = convolve(sdss_flux, gaussian_kernel)
Gaus_smoothed_DESI = convolve(flux_desi, gaussian_kernel)

H_alpha = 6562.7
H_beta = 4861.35
C3 = 5696
C4 = 1548
Mg2 = 2797
z = 0.385

#Plot of SDSS & DESI Spectra
plt.figure(figsize=(18,6))
plt.plot(wavelength_desi, flux_desi, alpha = 0.2, color = 'blue')
# plt.plot(wavelength_desi, Gaus_smoothed_DESI, color = 'blue', label = 'DESI')
plt.plot(wavelength_desi, DESI_rolling, color = 'blue', label = 'DESI')
plt.plot(sdss_measured_wl, sdss_flux, alpha = 0.2, color = 'orange')
# plt.plot(sdss_measured_wl, Gaus_smoothed_SDSS, color = 'orange', label = 'SDSS')
plt.plot(sdss_measured_wl, SDSS_rolling, color = 'orange', label = 'SDSS')
plt.axvline(H_alpha*(1+z), linewidth=2, color='red', label = 'H alpha')
plt.axvline(H_beta*(1+z), linewidth=2, color='red', label = 'H beta')
plt.axvline(C3*(1+z), linewidth=2, color='red', label = 'C [iii')
plt.axvline(C4*(1+z), linewidth=2, color='red', label = 'C iv')
plt.axvline(Mg2*(1+z), linewidth=2, color='red', label = 'Mg ii')
plt.xlabel('Wavelength / Å')
plt.ylabel('Flux / 10-17 ergs/s/cm2/Å')
# plt.title('Gaussian Smoothed Plot of SDSS & DESI Spectra')
plt.title('Manually Smoothed Plot of SDSS & DESI Spectra')
plt.legend(loc = 'upper right')
plt.show()

#Starting to do some fitting:
spec = Spectrum1D(spectral_axis=units_sdss_measured_wl, flux=units_sdss_flux)
# This line creates a figure (f) and an axis (ax) object using matplotlib.pyplot.subplots(). These objects will be used to generate the plot.
# f: The figure that contains the plot.
# ax: The specific axis on which the plot will be drawn
f, ax = plt.subplots()
# # This plots the data so the values change in discrete steps rather than a continuous line.
# ax.step(spec.spectral_axis, spec.flux, color='orange')
# plt.show()

import warnings
from specutils.fitting import fit_generic_continuum
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    cont_norm_spec = spec / fit_generic_continuum(spec)(spec.spectral_axis)

ax.step(cont_norm_spec.wavelength, cont_norm_spec.flux)  
ax.set_xlim(654 * u.nm, 660 * u.nm)

from specutils import SpectralRegion
from specutils.analysis import equivalent_width
equivalent_width(cont_norm_spec, regions=SpectralRegion(6562 * u.AA, 6575 * u.AA)) #vary depending on which line width you wish to look at

#Now I want to recreate some plots from the LATEST Guo data.
#I have access to table 4, so I will recreate figure 1:
#Specifically I will recrate the histograms from the bottom panels of figure 1.
#For the middle panel, I will use Topcat to recreate the grey plot (I don't have access to which group each CLAGN is in).
#I won't recreate the top panel as I don't have access to the parent sample data.

# table_4_GUO = pd.read_csv('guo23_table4_clagn.csv')

# # Step 2: Filter the data based on the 'transition Line' column
# turn_on_z = table_4_GUO[table_4_GUO['transition'] == 'turn-on']['Redshift']
# turn_off_z = table_4_GUO[table_4_GUO['transition'] == 'turn-off']['Redshift']
# # Includes CLAGN with more than 1 BEL

# # Step 3: Create the histogram
# plt.figure(figsize=(10, 6))

# my_list_x_axis_z = np.arange(0, 3.0, 0.2).tolist()

# # Plot histogram for 'turn-on'
# plt.hist(turn_on_z, bins=my_list_x_axis_z, histtype='step', label='Turn-on', color='blue')

# # Plot histogram for 'turn-off'
# plt.hist(turn_off_z, bins=my_list_x_axis_z, histtype='step', label='Turn-off', color='red')

# # Adding labels and title
# plt.xticks(my_list_x_axis_z)
# plt.xlim(0)  # Sets the minimum x-axis value to 0
# plt.xlabel('Redshift')
# plt.ylabel('N')
# plt.title('CLAGN Redshift: Turn-on vs Turn-off')
# plt.legend()

# # Show the plot
# plt.show()

# # Same plot but for r_band magnitude
# turn_on_r_mag = table_4_GUO[table_4_GUO['transition'] == 'turn-on']['r(mag)']
# turn_off_r_mag = table_4_GUO[table_4_GUO['transition'] == 'turn-off']['r(mag)']

# # Step 3: Create the histogram
# plt.figure(figsize=(20, 6))

# my_list_x_axis_r_mag = np.arange(18, 22.6, 0.2).tolist()

# # Plot histogram for 'turn-on'
# plt.hist(turn_on_r_mag, bins=my_list_x_axis_r_mag, histtype='step', label='Turn-on', color='blue')

# # Plot histogram for 'turn-off'
# plt.hist(turn_off_r_mag, bins=my_list_x_axis_r_mag, histtype='step', label='Turn-off', color='red')

# # Adding labels and title
# plt.xticks(my_list_x_axis_r_mag)
# plt.xlim(18)
# plt.xlabel('Magnitude (r-band)')
# plt.ylabel('N')
# plt.title('CLAGN r-band Magnitude: Turn-on vs Turn-off')
# plt.legend()

# # Show the plot
# plt.show()

# table_4_GUO['MJD_1'] = table_4_GUO['MJD_1'].fillna(method='ffill')
# table_4_GUO['MJD_2'] = table_4_GUO['MJD_2'].fillna(method='ffill')
# table_4_GUO['Redshift'] = table_4_GUO['Redshift'].fillna(method='ffill')

# table_4_GUO['Velocity Parallel'] = c*(((1+table_4_GUO['Redshift'])**2)-1)/(((1+table_4_GUO['Redshift'])**2)+1)

# table_4_GUO['Obs_MJD_diff'] = table_4_GUO['MJD_2'] - table_4_GUO['MJD_1']
# table_4_GUO['Rest_MJD_diff'] = (table_4_GUO['MJD_2'] - table_4_GUO['MJD_1'])*(np.sqrt(1-(((table_4_GUO['Velocity Parallel'])**2)/(c**2))))

# Obs_Time_scale_Halpha = table_4_GUO[table_4_GUO['Line'] == 'Halpha']['Obs_MJD_diff']
# Obs_Time_scale_Hbeta = table_4_GUO[table_4_GUO['Line'] == 'Hbeta']['Obs_MJD_diff']
# Obs_Time_scale_H = np.append(Obs_Time_scale_Halpha, Obs_Time_scale_Hbeta)
# Obs_Time_scale_C3 = table_4_GUO[table_4_GUO['Line'] == 'C iii]']['Obs_MJD_diff']
# Obs_Time_scale_C4 = table_4_GUO[table_4_GUO['Line'] == 'C iv']['Obs_MJD_diff']
# Obs_Time_scale_C = np.append(Obs_Time_scale_C3, Obs_Time_scale_C4)
# Obs_Time_scale_Mg2 = table_4_GUO[table_4_GUO['Line'] == 'Mg ii']['Obs_MJD_diff']
# Obs_Time_scale_Mg2.tolist()

# #Medians
# median_Obs_Time_scale_H = np.median(sorted(Obs_Time_scale_H))
# median_Obs_Time_scale_C = np.median(sorted(Obs_Time_scale_C))
# median_Obs_Time_scale_Mg2 = np.median(sorted(Obs_Time_scale_Mg2))

# plt.figure(figsize=(20, 6))

# my_list_x_axis_Obs_Timescale = np.arange(0, 8000, 500).tolist()

# plt.hist(Obs_Time_scale_C, bins=my_list_x_axis_Obs_Timescale, histtype='step', label='C III] + C IV', color='goldenrod')
# plt.axvline(median_Obs_Time_scale_C, linestyle='--', linewidth=2, color='goldenrod')
# plt.hist(Obs_Time_scale_Mg2, bins=my_list_x_axis_Obs_Timescale, histtype='step', label='MG II', color='green')
# plt.axvline(median_Obs_Time_scale_Mg2, linestyle='--', linewidth=2, color='green')
# plt.hist(Obs_Time_scale_H, bins=my_list_x_axis_Obs_Timescale, histtype='step', label='Halpha + Hbeta', color='blue')
# plt.axvline(median_Obs_Time_scale_H, linestyle='--', linewidth=2, color='blue')

# # Adding labels and title
# plt.xticks(my_list_x_axis_Obs_Timescale)
# plt.xlim(0)
# plt.ylim(0, 20)
# plt.xlabel('Timescale (days)')
# plt.ylabel('N')
# plt.title('CLAGN Observed Frame Timescale')
# plt.legend()

# plt.show()

# Rest_Time_scale_Halpha = table_4_GUO[table_4_GUO['Line'] == 'Halpha']['Rest_MJD_diff']
# Rest_Time_scale_Hbeta = table_4_GUO[table_4_GUO['Line'] == 'Hbeta']['Rest_MJD_diff']
# Rest_Time_scale_H = np.append(Rest_Time_scale_Halpha, Rest_Time_scale_Hbeta)
# Rest_Time_scale_C3 = table_4_GUO[table_4_GUO['Line'] == 'C iii]']['Rest_MJD_diff']
# Rest_Time_scale_C4 = table_4_GUO[table_4_GUO['Line'] == 'C iv']['Rest_MJD_diff']
# Rest_Time_scale_C = np.append(Rest_Time_scale_C3, Rest_Time_scale_C4)
# Rest_Time_scale_Mg2 = table_4_GUO[table_4_GUO['Line'] == 'Mg ii']['Rest_MJD_diff']
# Rest_Time_scale_Mg2.tolist()

# median_Rest_Time_scale_H = np.median(sorted(Rest_Time_scale_H))
# median_Rest_Time_scale_C = np.median(sorted(Rest_Time_scale_C))
# median_Rest_Time_scale_Mg2 = np.median(sorted(Rest_Time_scale_Mg2))

# plt.figure(figsize=(20, 6))

# my_list_x_axis_Rest_Timescale = np.arange(0, 7000, 250).tolist()

# plt.hist(Rest_Time_scale_C, bins=my_list_x_axis_Rest_Timescale, histtype='step', label='C III] + C IV', color='goldenrod')
# plt.axvline(median_Rest_Time_scale_C, linestyle='--', linewidth=2, color='goldenrod')
# plt.hist(Rest_Time_scale_Mg2, bins=my_list_x_axis_Rest_Timescale, histtype='step', label='MG II', color='green')
# plt.axvline(median_Rest_Time_scale_Mg2, linestyle='--', linewidth=2, color='green')
# plt.hist(Rest_Time_scale_H, bins=my_list_x_axis_Rest_Timescale, histtype='step', label='Halpha + Hbeta', color='blue')
# plt.axvline(median_Rest_Time_scale_H, linestyle='--', linewidth=2, color='blue')

# # Adding labels and title
# plt.xticks(my_list_x_axis_Rest_Timescale)
# plt.xlim(0)
# plt.ylim(0, 20)
# plt.xlabel('Timescale (days)')
# plt.ylabel('N')
# plt.title('CLAGN Rest Frame Timescale')
# plt.legend()

# plt.show()