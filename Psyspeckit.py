import numpy as np
import pyspeckit

# #Fitting a continuum model as a model
# xaxis = np.linspace(-50,150,100)
# sigma = 10.
# center = 50.

# baseline = np.poly1d([0.1, 0.25])(xaxis)

# synth_data = np.exp(-(xaxis-center)**2/(sigma**2 * 2.)) + baseline

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

# sp.specfit.Registry.add_fitter('polycontinuum',
#                                pyspeckit.models.polynomial_continuum.poly_fitter(),
#                                2)

# sp.specfit(fittype='polycontinuum', guesses=(0,0), exclude=[30, 70])

# # subtract the model fit to create a new spectrum
# sp_contsub = sp.copy()
# sp_contsub.data -= sp.specfit.get_full_model()
# sp_contsub.plotter()

# # Fit with automatic guesses
# sp_contsub.specfit(fittype='gaussian')

# # Fit with input guesses
# # The guesses initialize the fitter
# # This approach uses the 0th, 1st, and 2nd moments
# data = sp_contsub.data
# amplitude_guess = data.max()
# center_guess = (data*xaxis).sum()/data.sum()
# width_guess = (data.sum() / amplitude_guess / (2*np.pi))**0.5
# guesses = [amplitude_guess, center_guess, width_guess]
# sp_contsub.specfit(fittype='gaussian', guesses=guesses)

# sp_contsub.plotter(errstyle='fill')
# sp_contsub.specfit.plot_fit()

# Rest wavelengths of the lines we are fitting - use as initial guesses
NIIa = 6549.86
NIIb = 6585.27
Halpha = 6564.614
SIIa = 6718.29
SIIb = 6732.68

# Initialize spectrum object and plot region surrounding Halpha-[NII] complex
spec = pyspeckit.Spectrum('spec-8521-58175-0279.fits', errorcol=2)
spec.plotter(xmin = 6450, xmax = 6775, ymin = 0, ymax = 150)

# We fit the [NII] and [SII] doublets, and allow two components for Halpha.
# The widths of all narrow lines are tied to the widths of [SII].
guesses = [50, NIIa, 5, 100, Halpha, 5, 50, Halpha, 50, 50, NIIb, 5, 20, SIIa,
           5, 20, SIIb, 5]
tied = ['', '', 'p[17]', '', '', 'p[17]', '', 'p[4]', '', '3 * p[0]', '',
        'p[17]', '', '', 'p[17]', '', '', '']

# Actually do the fit.
spec.specfit(guesses = guesses, tied = tied, annotate = False)
spec.plotter.refresh()

# Let's use the measurements class to derive information about the emission
# lines.  The galaxy's redshift and the flux normalization of the spectrum
# must be supplied to convert measured fluxes to line luminosities.  If the
# spectrum we loaded in FITS format, 'BUNITS' would be read and we would not
# need to supply 'fluxnorm'. As is the case here
spec.measure(z = 0.385)

# Now overplot positions of lines and annotate

y = spec.plotter.ymax * 0.85    # Location of annotations in y

for i, line in enumerate(spec.measurements.lines.keys()):

    # If this line is not in our database of lines, don't try to annotate it
    if line not in spec.speclines.optical.lines.keys(): continue

    x = spec.measurements.lines[line]['modelpars'][1]   # Location of the emission line
    # Draw dashed line to mark its position
    spec.plotter.axis.plot([x]*2, [spec.plotter.ymin, spec.plotter.ymax],
                           ls='--', color='k')
    # Label it
    spec.plotter.axis.annotate(spec.speclines.optical.lines[line][-1], (x, y),
                               rotation = 90, ha = 'right', va = 'center')
# Make some nice axis labels
spec.plotter.axis.set_xlabel(r'Wavelength $(\AA)$')
spec.plotter.axis.set_ylabel(r'Flux $(10^{-17} \mathrm{erg/s/cm^2/\AA})$')
spec.plotter.refresh()

# Print out spectral line information
print("Line   Flux (erg/s/cm^2)     Amplitude (erg/s/cm^2)"
      "    FWHM (Angstrom)   Luminosity (erg/s)")
for line in spec.measurements.lines.keys():
    print(line, spec.measurements.lines[line]['flux'],
          spec.measurements.lines[line]['amp'],
          spec.measurements.lines[line]['fwhm'],
          spec.measurements.lines[line]['lum'])

# Had we not supplied the objects redshift (or distance), the line
# luminosities would not have been measured, but integrated fluxes would
# still be derived.  Also, the measurements class separates the broad and
# narrow H-alpha components, and identifies which lines are which. How nice!

spec.specfit.plot_fit()