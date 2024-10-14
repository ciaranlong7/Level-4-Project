import numpy as np
import matplotlib.pyplot as plt
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