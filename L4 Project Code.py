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

#Open the SDSS file
with fits.open("spec-8521-58175-0279.fits") as hdul:
    subset = hdul[1]       

    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
    units_sdss_flux = subset.data['flux'] * 10**-17 * u.Unit('erg cm-2 s-1 AA-1') #for fitting
    sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
    units_sdss_lamb = 10**subset.data['loglam'] * u.AA 
    sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])

#Open the DESI file
DESI_spec = pd.read_csv('spectrum_desi_152517.57+401357.6.csv')
desi_lamb = DESI_spec.iloc[1:, 0]  # First column, skipping the first row (header)
desi_flux = DESI_spec.iloc[1:, 1]  # Second column, skipping the first row (header)

# Correcting for redshift.
z = 0.385
sdss_lamb = sdss_lamb/(1+z)
desi_lamb = desi_lamb/(1+z)
units_sdss_lamb = units_sdss_lamb/(1+z)

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
DESI_rolling = rolling_average(desi_flux, 10)
sdss_lamb = sdss_lamb[9:]
desi_lamb = desi_lamb[9:]
sdss_flux = sdss_flux[9:]
desi_flux = desi_flux[9:]

# Gaussian smoothing
# adjust stddev to control the degree of smoothing. Higher stddev means smoother
# https://en.wikipedia.org/wiki/Gaussian_blur
gaussian_kernel = Gaussian1DKernel(stddev=3)

# Smooth the flux data using the Gaussian kernel
Gaus_smoothed_SDSS = convolve(sdss_flux, gaussian_kernel)
Gaus_smoothed_DESI = convolve(desi_flux, gaussian_kernel)

H_alpha = 6562.7
H_beta = 4861.35
# C3 = UV line
C4 = 1548
Mg2 = 2797

#Plot of SDSS & DESI Spectra
plt.figure(figsize=(18,6))
plt.plot(desi_lamb, desi_flux, alpha = 0.2, color = 'blue')
plt.plot(desi_lamb, Gaus_smoothed_DESI, color = 'blue', label = 'DESI')
# plt.plot(desi_lamb, DESI_rolling, color = 'blue', label = 'DESI')
plt.plot(sdss_lamb, sdss_flux, alpha = 0.2, color = 'orange')
plt.plot(sdss_lamb, Gaus_smoothed_SDSS, color = 'orange', label = 'SDSS')
# plt.plot(sdss_lamb, SDSS_rolling, color = 'orange', label = 'SDSS')
plt.axvline(H_alpha, linewidth=2, color='goldenrod', label = 'H alpha')
plt.axvline(H_beta, linewidth=2, color='green', label = 'H beta')
plt.axvline(Mg2, linewidth=2, color='red', label = 'Mg ii')
plt.xlabel('Wavelength / Å')
plt.ylabel('Flux / 10-17 ergs/s/cm2/Å')
plt.title('Gaussian Smoothed Plot of SDSS & DESI Spectra')
# plt.title('Manually Smoothed Plot of SDSS & DESI Spectra')
plt.legend(loc = 'upper right')
# plt.show()

# #Starting to do some fitting:
filename = 'https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/1323/spec-1323-52797-0012.fits'
# The spectrum is in the second HDU of this file.
with fits.open(filename) as f:
    specdata = f[1].data

lamb = 10**specdata['loglam'] * u.AA 
flux = specdata['flux'] * 10**-17 * u.Unit('erg cm-2 s-1 AA-1') 

spec = Spectrum1D(spectral_axis=units_sdss_lamb, flux=units_sdss_flux)
# spec = Spectrum1D(spectral_axis=lamb, flux=flux)

# print(np.isnan(units_sdss_lamb).sum())
# print(np.isnan(units_sdss_flux).sum())
# print(np.isnan(lamb).sum())
# print(np.isnan(flux).sum())

print(spec)
print(Spectrum1D(spectral_axis=lamb, flux=flux))

f, ax = plt.subplots()  
ax.step(spec.spectral_axis, spec.flux)


# "Now maybe you want the equivalent width of a spectral line. That requires normalizing by a continuum estimate:""
import warnings
from specutils.fitting import fit_generic_continuum
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    cont_norm_spec = spec/fit_generic_continuum(spec)(spec.spectral_axis)

#plot spectra theyre using. see what it looks like. Test what cases the spectra fitting works.

print(fit_generic_continuum(spec)(spec.spectral_axis))
#Problem is with above. I am printing the denominator, but it's just a list of 0s.

f, ax = plt.subplots()  
ax.step(cont_norm_spec.wavelength, cont_norm_spec.flux)  
# ax.set_xlim(654 * u.nm, 657.5 * u.nm)
plt.show()

from specutils import SpectralRegion
from specutils.analysis import equivalent_width
equivalent_width(cont_norm_spec, regions=SpectralRegion(6540 * u.AA, 6575 * u.AA))

#Plotting MIR data
def flux(mag, k): # k is the zero magnitude flux density. Taken from a table
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys
W2_k = 171.787

MIR_data = pd.read_csv('Object_MIR_data.csv')

# Filter the DataFrame for rows where cc_flags is 0
filtered_rows = MIR_data[MIR_data.iloc[:, 15] == 0]

# Extract W1 & W2 mag from the w1mpro & w2mpro columns (index 5, 9) of the filtered rows
W1_mag = filtered_rows.iloc[:, 5]
W1_mag = W1_mag.tolist()
W2_mag = filtered_rows.iloc[:, 9]
W2_mag = W2_mag.tolist()
mjd_date = filtered_rows.iloc[:, 18]
mjd_date = mjd_date.tolist()

colour = []
for i in range(len(W1_mag)):
    colour.append(W2_mag[i] - W1_mag[i])

plt.figure(figsize=(18,6))
plt.scatter(mjd_date, W1_mag, color = 'orange', label = r'W1 (3.4 \u03bcm)')
plt.scatter(mjd_date, W2_mag, color = 'blue', label = r'W2 (4.6 \u03bcm)')
# plt.scatter(mjd_date, colour, color = 'red', label = r'Colour (W2 Flux - W1 Flux)')
plt.xlabel('Date / mjd date')
plt.ylabel('Magnitude')
plt.title('W1, W2 vs time')
plt.legend(loc = 'upper right')
plt.show()

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