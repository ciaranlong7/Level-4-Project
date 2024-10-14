import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

with fits.open("spec-8521-58175-0279.fits") as hdul:
    subset = hdul[1]        

    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
    sdss_measured_wl = 10**subset.data['loglam'] #Wavelength in Angstroms
    sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])

#Plotting sdss flux against wavelength
plt.plot(sdss_measured_wl, sdss_flux, label = 'Spectrum')
plt.xlabel('Wavelength / Å', fontsize = 16)
plt.ylabel('Flux', fontsize = 16)
plt.legend(loc = 'upper right')
plt.show()

#Open the CSV file using pandas
df = pd.read_csv('spectrum_desi_152517.57+401357.6.csv')

#Extract the first and second columns
wavelength_desi = df.iloc[1:, 0]  # First column, skipping the first row (header)
flux_desi = df.iloc[1:, 1]  # Second column, skipping the first row (header)

plt.plot(wavelength_desi, flux_desi, label = 'Desi')
plt.xlabel('Wavelength / Å')
plt.ylabel('Flux')
plt.show()