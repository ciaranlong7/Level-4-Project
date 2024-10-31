import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from specutils import Spectrum1D
from astropy.visualization import quantity_support
quantity_support()  # for getting units on the axes below

#Open the SDSS file
with fits.open("spec-8521-58175-0279.fits") as hdul:
    subset = hdul[1]       

    units_sdss_flux = subset.data['flux'] * 10**-17 * u.Unit('erg cm-2 s-1 AA-1') #for fitting
    units_sdss_lamb = 10**subset.data['loglam'] * u.AA 

filename = 'https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/1323/spec-1323-52797-0012.fits'
# The spectrum is in the second HDU of this file.
with fits.open(filename) as f:
    specdata = f[1].data

lamb = 10**specdata['loglam'] * u.AA 
flux = specdata['flux'] * 10**-17 * u.Unit('erg cm-2 s-1 AA-1') 

spec = Spectrum1D(spectral_axis=units_sdss_lamb, flux=units_sdss_flux)
#spec = Spectrum1D(spectral_axis=lamb, flux=flux)

print(spec)
print(Spectrum1D(spectral_axis=lamb, flux=flux))

f, ax = plt.subplots()  
ax.step(spec.spectral_axis, spec.flux)
#simplify script. dont ignore warnings. can go to line/spectrum fitting.
# send claire my script maybe if i still cant find.

# "Now maybe you want the equivalent width of a spectral line. That requires normalizing by a continuum estimate:""
import warnings
from specutils.fitting import fit_generic_continuum
with warnings.catch_warnings():  # Ignore warnings
    warnings.simplefilter('ignore')
    cont_norm_spec = spec/fit_generic_continuum(spec)(spec.spectral_axis)

#plot spectra theyre using. see what it looks like. Test what cases the spectra fitting works.

# print(fit_generic_continuum(spec)(spec.spectral_axis))
#Problem is with above. I am printing the denominator, but it's just a list of 0s.

f, ax = plt.subplots()  
ax.step(cont_norm_spec.wavelength, cont_norm_spec.flux)  
ax.set_xlim(654 * u.nm, 657.5 * u.nm)
plt.show()

from specutils import SpectralRegion
from specutils.analysis import equivalent_width
equivalent_width(cont_norm_spec, regions=SpectralRegion(6540 * u.AA, 6575 * u.AA))