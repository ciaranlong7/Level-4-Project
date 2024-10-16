import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits

#Open the SDSS file
with fits.open("spec-8521-58175-0279.fits") as hdul:
    subset = hdul[1]        

    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
    sdss_measured_wl = 10**subset.data['loglam'] #Wavelength in Angstroms
    sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])

#Open the DESI file
df = pd.read_csv('spectrum_desi_152517.57+401357.6.csv')
wavelength_desi = df.iloc[1:, 0]  # First column, skipping the first row (header)
flux_desi = df.iloc[1:, 1]  # Second column, skipping the first row (header)

#Plot of SDSS & DESI Spectra
plt.figure(figsize=(18,6))
plt.plot(wavelength_desi, flux_desi, label = 'DESI')
plt.plot(sdss_measured_wl, sdss_flux, label = 'SDSS')
plt.xlabel('Wavelength / Å')
plt.ylabel('Flux / ?')
plt.title('Plot of SDSS & DESI Spectra')
plt.legend(loc = 'upper right')
plt.show()