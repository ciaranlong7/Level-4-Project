import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import scipy.optimize
from astropy.io import fits
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.coordinates import SkyCoord
import sfdmap
from astroquery.ipac.irsa import Irsa
from dust_extinction.parameter_averages import G23
from astropy.io.fits.hdu.hdulist import HDUList
from astroquery.sdss import SDSS
from sparcl.client import SparclClient
from dl import queryClient as qc
# import os
# import fitsio
# import desispec.io

c = 299792458

# Plotting ideas:
# Find a way to convert from SDSS & DESI flux to mag.
# Look into spectral fitting of DESI & SDSS spectra.

#G23 dust extinction model:
#https://dust-extinction.readthedocs.io/en/latest/api/dust_extinction.parameter_averages.G23.html#dust_extinction.parameter_averages.G23

#get SDSS & DESI filenames:
# object_name = '152517.57+401357.6' #Object A - assigned to me
# object_name = '141923.44-030458.7' #Object B - chosen because of very high redshift
# object_name = '115403.00+003154.0' #Object C - randomly chosen, but it had a low redshift also
# object_name = '140957.72-012850.5' #Object D - chosen because of very high z scores
# object_name = '162106.25+371950.7' #Object E - chosen because of very low z scores
object_name = '135544.25+531805.2' #Object F - chosen because not a CLAGN, but in AGN parent sample & has high z scores
# object_name = '150210.72+522212.2' #Object G - chosen because not a CLAGN, but in AGN parent sample & has low z scores
# object_name = '101536.17+221048.9' #Highly variable AGN object 1 (no SDSS reading in parent sample)
# object_name = '090931.55-011233.3' #Highly variable AGN object 2 (no SDSS reading in parent sample)
# object_name = '151639.06+280520.4' #Object H - chosen because not a CLAGN, but in AGN parent sample & has high z scores & normalised flux change
# object_name = '160833.97+421413.4' #Object I - chosen because not a CLAGN, but in AGN parent sample & has high normalised flux change
# object_name = '164837.68+311652.7' #Object J - chosen because not a CLAGN, but in AGN parent sample & has high z scores

def flux(mag, k, wavel): # k is the zero magnitude flux density. For W1 & W2, taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
        k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
        return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys. This means that mag 0 = 309.540 Janskys at the W1 wl.
W2_k = 171.787
g_k = 3991
r_k = 3174
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4
g_wl = 0.467e4
r_wl = 0.616e4

Min_SNR = 3 #Options are 10, 3, or 2. #A (SNR>10), B (3<SNR<10) or C (2<SNR<3)
if Min_SNR == 10: #Select Min_SNR on line above.
    MIR_SNR = 'A'
elif Min_SNR == 3:
    MIR_SNR = 'B'
elif Min_SNR == 2:
    MIR_SNR = 'C'
else:
    print('select a valid min SNR - 10, 3 or 2.')
parent_sample = pd.read_csv('guo23_parent_sample.csv')
object_data = parent_sample[parent_sample.iloc[:, 4] == object_name]
SDSS_RA = object_data.iloc[0, 1]
SDSS_DEC = object_data.iloc[0, 2]
SDSS_plate_number = object_data.iloc[0, 5]
SDSS_fiberid_number = object_data.iloc[0, 7]
SDSS_mjd = object_data.iloc[0, 6]
DESI_mjd = object_data.iloc[0, 12]
SDSS_z = object_data.iloc[0, 3]
DESI_z = object_data.iloc[0, 10]
DESI_name = object_data.iloc[0, 11]

# print('MIR Search (RA ±DEC):')
# print(f'{SDSS_RA} {SDSS_DEC:+}')

#Automatically querying the SDSS database
downloaded_SDSS_spec = SDSS.get_spectra_async(plate=SDSS_plate_number, fiberID=SDSS_fiberid_number, mjd=SDSS_mjd)
downloaded_SDSS_spec = downloaded_SDSS_spec[0]

hdul = HDUList(downloaded_SDSS_spec.get_fits())
subset = hdul[1]

#SDSS Spectrum information
sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])

#DESI spectrum retrieval method
# from - https://github.com/astro-datalab/notebooks-latest/blob/master/03_ScienceExamples/DESI/01_Intro_to_DESI_EDR.ipynb
def get_primary_spectrum(targetid): #some objects have multiple spectra for it in DESI- the best one is the 'primary' spectrum
    """
    Retrieves the primary spectrum's wavelength and flux for a given target ID.

    Parameters:
    - targetid (int): The target ID of the object.

    Returns:
    - lam_primary (array): Wavelength data for the primary spectrum.
    - flam_primary (array): Flux data for the primary spectrum.
    """
    # Initialize SparclClient
    client = SparclClient()
    
    # Get all available fields
    inc = client.get_all_fields()

    # Retrieve the spectrum by target ID
    res = client.retrieve_by_specid(specid_list=[targetid], include=inc, dataset_list=['DESI-EDR'])
    # res = client.retrieve_by_specid(specid_list=[targetid], include=inc, dataset_list=['DESI-EDR', 'DESI-DR1']) # only have access to EDR

    # Extract records
    records = res.records

    if not records: #no spectrum could be found:
        print(f'Spectrum cannot be found for object_name = {object_name}, DESI target_id = {DESI_name}')

        try:
            DESI_file = f'spectrum_desi_{object_name}.csv'
            DESI_file_path = f'clagn_spectra/{DESI_file}'
            DESI_spec = pd.read_csv(DESI_file_path)
            desi_lamb = DESI_spec.iloc[1:, 0]  # First column, skipping the first row (header)
            desi_flux = DESI_spec.iloc[1:, 1]  # Second column, skipping the first row (header)
            print('DESI file is in downloads - will proceed as normal')
            return desi_lamb, desi_flux
        except FileNotFoundError as e:
            print('No DESI file already downloaded.')
            return [], []

    # Identify the primary spectrum
    spec_primary = np.array([records[jj].specprimary for jj in range(len(records))])

    if not np.any(spec_primary):
        print(f'Spectrum cannot be found for object_name = {object_name}, DESI target_id = {DESI_name}')

        try:
            DESI_file = f'spectrum_desi_{object_name}.csv'
            DESI_file_path = f'clagn_spectra/{DESI_file}'
            DESI_spec = pd.read_csv(DESI_file_path)
            desi_lamb = DESI_spec.iloc[1:, 0]  # First column, skipping the first row (header)
            desi_flux = DESI_spec.iloc[1:, 1]  # Second column, skipping the first row (header)
            print('DESI file is in downloads - will proceed as normal')
            return desi_lamb, desi_flux
        except FileNotFoundError as e:
            print('No DESI file already downloaded.')
            return [], []

    # Get the index of the primary spectrum
    primary_ii = np.where(spec_primary == True)[0][0]

    # Extract wavelength and flux for the primary spectrum
    desi_lamb = records[primary_ii].wavelength
    desi_flux = records[primary_ii].flux

    return desi_lamb, desi_flux

target_id = int(DESI_name)
desi_lamb, desi_flux = get_primary_spectrum(target_id)

coord = SkyCoord(SDSS_RA, SDSS_DEC, unit='deg', frame='icrs') #This works

sfd = sfdmap.SFDMap('SFD_dust_files') #called SFD map, but see - https://github.com/kbarbary/sfdmap/blob/master/README.md
# It explains how "By default, a scaling of 0.86 is applied to the map values to reflect the recalibration by Schlafly & Finkbeiner (2011)"
ebv = sfd.ebv(coord)
print(f"E(B-V): {ebv}")

ext_model = G23(Rv=3.1) #Rv=3.1 is typical for MW - Schultz, Wiemer, 1975
# uncorrected_SDSS = sdss_flux
inverse_SDSS_lamb = [1/(x*10**(-4)) for x in sdss_lamb] #need units of inverse microns for extinguishing
inverse_DESI_lamb = [1/(x*10**(-4)) for x in desi_lamb]
sdss_flux = sdss_flux/ext_model.extinguish(inverse_SDSS_lamb, Ebv=ebv) #divide to remove the effect of dust
desi_flux = desi_flux/ext_model.extinguish(inverse_DESI_lamb, Ebv=ebv)

# Correcting for redshift.
sdss_lamb = (sdss_lamb/(1+DESI_z))
desi_lamb = (desi_lamb/(1+DESI_z))

#Calculate rolling average manually
def rolling_average(arr, window_size):
    
    averages = []
    for i in range(len(arr) - window_size + 1):
        avg = np.mean(arr[i:i + window_size])
        averages.append(avg)
    return np.array(averages)

#Manual Rolling averages - only uncomment if using (otherwise cuts off first 9 data points)
# SDSS_rolling = rolling_average(sdss_flux, 10)
# DESI_rolling = rolling_average(desi_flux, 10)
# sdss_lamb = sdss_lamb[9:]
# desi_lamb = desi_lamb[9:]
# sdss_flux = sdss_flux[9:]
# desi_flux = desi_flux[9:]

# Gaussian smoothing
# adjust stddev to control the degree of smoothing. Higher stddev means smoother
# https://en.wikipedia.org/wiki/Gaussian_blur
gaussian_kernel = Gaussian1DKernel(stddev=3)

# Smooth the flux data using the Gaussian kernel
Gaus_smoothed_SDSS = convolve(sdss_flux, gaussian_kernel)
if len(desi_flux) > 0:
    Gaus_smoothed_DESI = convolve(desi_flux, gaussian_kernel)
else:
    Gaus_smoothed_DESI = []
# Gaus_smoothed_SDSS_uncorrected = convolve(uncorrected_SDSS, gaussian_kernel)

#BELs
H_alpha = 6562.819
H_beta = 4861.333
Mg2 = 2795.528
C3_ = 1908.734
C4 = 1548.187
Ly_alpha = 1215.670
Ly_beta = 1025.722
#NEL
_O3_ = 5006.843 #underscores indicate square brackets
#Note there are other [O III] lines, such as: 4958.911 A, 4363.210 A
SDSS_min = min(sdss_lamb)
SDSS_max = max(sdss_lamb)
if len(desi_lamb) > 0:
    DESI_min = min(desi_lamb)
    DESI_max = max(desi_lamb)
else:
    DESI_min = 0
    DESI_max = 1

# #Plot of SDSS Spectrum - Extinction Corrected vs Uncorrected
# plt.figure(figsize=(12,7))
# plt.plot(sdss_lamb, Gaus_smoothed_SDSS, color = 'orange', label = 'Extinction Corrected')
# plt.plot(sdss_lamb, Gaus_smoothed_SDSS_uncorrected, color = 'blue', label = 'Uncorrected')
# plt.xlabel('Wavelength / Å')
# plt.ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# plt.title('SDSS Spectrum - Extinction Corrected vs Uncorrected')
# plt.legend(loc = 'upper right')
# plt.show()

# # #Plot of SDSS Spectrum with uncertainties
# plt.figure(figsize=(12,7))
# plt.errorbar(sdss_lamb, sdss_flux, yerr=sdss_flux_unc, fmt='o', color = 'forestgreen', capsize=5)
# plt.xlabel('Wavelength / Å')
# plt.ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# plt.title(f'SDSS Spectrum {object_name}')
# plt.show()

# # Plot of SDSS & DESI Spectra
# plt.figure(figsize=(12,7))
# # Original unsmoothed spectrum
# plt.plot(sdss_lamb, sdss_flux, alpha = 0.2, color = 'forestgreen')
# plt.plot(desi_lamb, desi_flux, alpha = 0.2, color = 'midnightblue')
# # Gausian smoothing
# plt.plot(sdss_lamb, Gaus_smoothed_SDSS, color = 'forestgreen', label = 'SDSS')
# plt.plot(desi_lamb, Gaus_smoothed_DESI, color = 'midnightblue', label = 'DESI')
# # # Manual smoothing
# # plt.plot(sdss_lamb, SDSS_rolling, color = 'forestgreen', label = 'SDSS')
# # plt.plot(desi_lamb, DESI_rolling, color = 'midnightblue', label = 'DESI')
# # # Adding in positions of emission lines
# if SDSS_min <= H_alpha <= SDSS_max:
#     plt.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
# if SDSS_min <= H_beta <= SDSS_max:
#     plt.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
# if SDSS_min <= Mg2 <= SDSS_max:
#     plt.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
# if SDSS_min <= C3_ <= SDSS_max:
#     plt.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
# if SDSS_min <= C4 <= SDSS_max:
#     plt.axvline(C4, linewidth=2, color='violet', label = 'C IV')
# if SDSS_min <= _O3_ <= SDSS_max:
#     plt.axvline(_O3_, linewidth=2, color='grey', label = '[O III]')
# if SDSS_min <= Ly_alpha <= SDSS_max:
#     plt.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
# if SDSS_min <= Ly_beta <= SDSS_max:
#     plt.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
# # Axes labels
# plt.xlabel('Wavelength / Å')
# plt.ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# # Two different titles (for Gaussian/Manual)
# plt.title('Gaussian Smoothed Plot of SDSS & DESI Spectra')
# # plt.title('Manually Smoothed Plot of SDSS & DESI Spectra')
# plt.legend(loc = 'upper right')
# plt.show()

# Automatically querying catalogues
# coord = SkyCoord(153.9007071, 22.1802528, unit='deg', frame='icrs') #1st Highly Variable AGN - object_name = J101536.17+221048.9
WISE_query = Irsa.query_region(coordinates=coord, catalog="allwise_p3as_mep", spatial="Cone", radius=2 * u.arcsec)
NEOWISE_query = Irsa.query_region(coordinates=coord, catalog="neowiser_p1bs_psd", spatial="Cone", radius=2 * u.arcsec)
# PTF_query = Irsa.query_region(coordinates=coord, catalog="ptf_lightcurves", spatial="Cone", radius=2 * u.arcsec)
WISE_data = WISE_query.to_pandas()
NEO_data = NEOWISE_query.to_pandas()
# PTF_data = PTF_query.to_pandas()

# # # checking out indexes
# for idx, col in enumerate(WISE_data.columns):
#     print(f"Index {idx}: {col}")

# print(PTF_data.iloc[:, 3].unique()) #problem - there are two objects in my search
# print(WISE_data.iloc[:, 41].unique())

WISE_data = WISE_data.sort_values(by=WISE_data.columns[10]) #sort in ascending mjd
NEO_data = NEO_data.sort_values(by=NEO_data.columns[42]) #sort in ascending mjd
# PTF_data = PTF_data.sort_values(by=PTF_data.columns[0]) #sort in ascending mjd

WISE_data.iloc[:, 6] = pd.to_numeric(WISE_data.iloc[:, 6], errors='coerce')
filtered_WISE_rows = WISE_data[(WISE_data.iloc[:, 6] == 0) & (WISE_data.iloc[:, 39] == 1) & (WISE_data.iloc[:, 41] == '0000') & (WISE_data.iloc[:, 40] > 5)]
#filtering for cc_flags == 0 in all bands, qi_fact == 1, no moon masking flag & separation of the WISE instrument to the SAA > 5 degrees. Unlike with Neowise, there is no individual column for cc_flags in each band

filtered_NEO_rows = NEO_data[(NEO_data.iloc[:, 36] > 5) & (NEO_data.iloc[:, 38] > 5)] #checking for rows where qual_frame is > 5 & separation of the WISE instrument to the South Atlantic Anomaly is > 5 degrees
#"Single-exposure source database entries having qual_frame=0 should be used with extreme caution" - from the column descriptions.
# The qi_fact column seems to be equal to qual_frame/10.

# filtered_PTF_rows = PTF_data[(PTF_data.iloc[:, 34] == 1) & (PTF_data.iloc[:, 35] == 1)] #filtering for photcalflag == 1 (indicating source is photometrically calibrated) & goodflag == 1 (indicating source is good)

#Filtering for good SNR, no cc_flags & no moon scattering flux
if MIR_SNR == 'C':
    filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX', 'BA', 'BB', 'BC', 'BU', 'BX', 'CA', 'CB', 'CC', 'CU', 'CX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
    filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA', 'AB', 'BB', 'CB', 'UB', 'XB', 'AC', 'BC', 'CC', 'UC', 'XC'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]
if MIR_SNR == 'B':
    filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX', 'BA', 'BB', 'BC', 'BU', 'BX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
    filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA', 'AB', 'BB', 'CB', 'UB', 'XB'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]
if MIR_SNR == 'A':
    filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
    filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]

mjd_date_W1 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W1.iloc[:, 42].tolist()
W1_mag = filtered_WISE_rows.iloc[:, 11].tolist() + filtered_NEO_rows_W1.iloc[:, 18].tolist()
# W1_mag = filtered_WISE_rows.iloc[:, 23].tolist() + filtered_NEO_rows_W1.iloc[:, 53].tolist() # raw flux
W1_unc = filtered_WISE_rows.iloc[:, 12].tolist() + filtered_NEO_rows_W1.iloc[:, 19].tolist()
# W1_unc = filtered_WISE_rows.iloc[:, 24].tolist() + filtered_NEO_rows_W1.iloc[:, 54].tolist() #raw flux unc
W1_mag = list(zip(W1_mag, mjd_date_W1, W1_unc))
W1_mag = [tup for tup in W1_mag if not np.isnan(tup[0])] #removing instances where the mag value is NaN

mjd_date_W2 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W2.iloc[:, 42].tolist()
W2_mag = filtered_WISE_rows.iloc[:, 14].tolist() + filtered_NEO_rows_W2.iloc[:, 22].tolist()
# W2_mag = filtered_WISE_rows.iloc[:, 25].tolist() + filtered_NEO_rows_W1.iloc[:, 55].tolist()
W2_unc = filtered_WISE_rows.iloc[:, 15].tolist() + filtered_NEO_rows_W2.iloc[:, 23].tolist()
# W2_unc = filtered_WISE_rows.iloc[:, 26].tolist() + filtered_NEO_rows_W1.iloc[:, 56].tolist()
W2_mag = list(zip(W2_mag, mjd_date_W2, W2_unc))
W2_mag = [tup for tup in W2_mag if not np.isnan(tup[0])]

# filtered_PTF_rows_g = filtered_PTF_rows[filtered_PTF_rows.iloc[:, 6] == 1] #using filter identifier column to select g_band observations
# filtered_PTF_rows_r = filtered_PTF_rows[filtered_PTF_rows.iloc[:, 6] == 2]

# mjd_date_PTF_g = filtered_PTF_rows_g.iloc[:, 0].tolist()
# PTF_mag_g = filtered_PTF_rows_g.iloc[:, 1].tolist()
# PTF_unc_g = filtered_PTF_rows_g.iloc[:, 2].tolist()

# mjd_date_PTF_r = filtered_PTF_rows_r.iloc[:, 0].tolist()
# PTF_mag_r = filtered_PTF_rows_r.iloc[:, 1].tolist()
# PTF_unc_r = filtered_PTF_rows_r.iloc[:, 2].tolist()

print(f'Object Name = {object_name}')
print(f'SDSS Redshift = {SDSS_z}')
print(f'DESI Redshift = {DESI_z}')
print(f'W1 data points = {len(W1_mag)}')
print(f'W2 data points = {len(W2_mag)}')
# print(f'g data points = {len(PTF_mag_g)}')
# print(f'r data points = {len(PTF_mag_r)}')

#Object A - The four W1_mag dps with ph_qual C are in rows, 29, 318, 386, 388

#Below code sorts MIR data.
#One assumption required for code to work:
#1. The data is sorted in order of oldest mjd to most recent.

# W1 data first
W1_list = []
W1_unc_list = []
W1_mjds = []
W1_averages= []
W1_av_uncs = []
W1_av_mjd_date = []
one_epoch_W1 = []
one_epoch_W1_unc = []
m = 0 # Change depending on which epoch you wish to look at. m = 0 represents epoch 1. Causes error if (m+1)>number of epochs
p = 0
for i in range(len(W1_mag)):
    if i == 0: #first reading - store and move on
        W1_list.append(W1_mag[i][0])
        W1_mjds.append(W1_mag[i][1])
        W1_unc_list.append(W1_mag[i][2])
        continue
    elif i == len(W1_mag) - 1: #if final data point, close the epoch
        W1_list.append(W1_mag[i][0])
        W1_mjds.append(W1_mag[i][1])
        W1_unc_list.append(W1_mag[i][2])
        W1_averages.append(np.average(W1_list))
        W1_av_mjd_date.append(np.average(W1_mjds))
        W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
        if p == m:
            one_epoch_W1 = W1_list
            one_epoch_W1_unc = W1_unc_list
            mjd_value = W1_mag[i][1]
            p += 1
        p += 1
        continue
    elif W1_mag[i][1] - W1_mag[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
        W1_list.append(W1_mag[i][0])
        W1_mjds.append(W1_mag[i][1])
        W1_unc_list.append(W1_mag[i][2])
        continue
    else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
        W1_averages.append(np.average(W1_list))
        W1_av_mjd_date.append(np.average(W1_mjds))
        W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
        if p == m:
            one_epoch_W1 = W1_list
            one_epoch_W1_unc = W1_unc_list
            mjd_value = W1_mag[i][1]
            p += 1
        W1_list = []
        W1_mjds = []
        W1_unc_list = []
        W1_list.append(W1_mag[i][0])
        W1_mjds.append(W1_mag[i][1])
        W1_unc_list.append(W1_mag[i][2])
        p += 1
        continue

# W2 data second
W2_list = []
W2_unc_list = []
W2_mjds = []
W2_averages= []
W2_av_uncs = []
W2_av_mjd_date = []
one_epoch_W2 = []
one_epoch_W2_unc = []
m = 0 # Change depending on which epoch you wish to look at. m = 0 represents epoch 1. Causes error if (m+1)>number of epochs
p = 0
for i in range(len(W2_mag)):
    if i == 0: #first reading - store and move on
        W2_list.append(W2_mag[i][0])
        W2_mjds.append(W2_mag[i][1])
        W2_unc_list.append(W2_mag[i][2])
        continue
    elif i == len(W2_mag) - 1: #if final data point, close the epoch
        W2_list.append(W2_mag[i][0])
        W2_mjds.append(W2_mag[i][1])
        W2_unc_list.append(W2_mag[i][2])
        W2_averages.append(np.average(W2_list))
        W2_av_mjd_date.append(np.average(W2_mjds))
        W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
        if p == m:
            one_epoch_W2 = W2_list
            one_epoch_W2_unc = W2_unc_list
            mjd_value = W2_mag[i][1]
            p += 1
        p += 1
        continue
    elif W2_mag[i][1] - W2_mag[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
        W2_list.append(W2_mag[i][0])
        W2_mjds.append(W2_mag[i][1])
        W2_unc_list.append(W2_mag[i][2])
        continue
    else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
        W2_averages.append(np.average(W2_list))
        W2_av_mjd_date.append(np.average(W2_mjds))
        W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
        if p == m:
            one_epoch_W2 = W2_list
            one_epoch_W2_unc = W2_unc_list
            mjd_value = W2_mag[i][1]
            p += 1
        W2_list = []
        W2_mjds = []
        W2_unc_list = []
        W2_list.append(W2_mag[i][0])
        W2_mjds.append(W2_mag[i][1])
        W2_unc_list.append(W2_mag[i][2])
        p += 1
        continue

# #PTF averaging
# g_list = []
# g_unc_list = []
# g_av_mag = []
# g_av_uncs = []
# mjd_list_g = []
# mjd_date_g_epoch = []
# one_epoch_g = []
# one_epoch_g_unc = []
# m = 0 #select an epoch, for both g & r band
# p = 0
# for i in range(len(PTF_mag_g)):
#     if i == 0:
#         g_list.append(PTF_mag_g[i])
#         g_unc_list.append(PTF_unc_g[i])
#         mjd_list_g.append(mjd_date_PTF_g[i])
#         continue
#     elif i == len(PTF_mag_g) - 1: #if final data point, close the epoch
#         g_av_mag.append(np.average(g_list))
#         g_av_uncs.append((1/len(g_unc_list))*np.sqrt(np.sum(np.square(g_unc_list))))
#         mjd_date_g_epoch.append(np.average(mjd_list_g))
#         if p == m:
#             one_epoch_g = g_list
#             one_epoch_g_unc = g_unc_list
#             one_epoch_g_mjd = mjd_list_g
#             p += 1
#         continue
#     elif mjd_date_PTF_g[i] - mjd_date_PTF_g[i-1] < 100:
#         g_list.append(PTF_mag_g[i])
#         g_unc_list.append(PTF_unc_g[i])
#         mjd_list_g.append(mjd_date_PTF_g[i])
#         continue
#     else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
#         g_av_mag.append(np.average(g_list))
#         g_av_uncs.append((1/len(g_unc_list))*np.sqrt(np.sum(np.square(g_unc_list))))
#         mjd_date_g_epoch.append(np.average(mjd_list_g))
#         if p == m:
#             one_epoch_g = g_list
#             one_epoch_g_unc = g_unc_list
#             one_epoch_g_mjd = mjd_list_g
#             p += 1
#         g_list = []
#         g_unc_list = []
#         mjd_list_g = []
#         g_list.append(PTF_mag_g[i])
#         g_unc_list.append(PTF_unc_g[i])
#         mjd_list_g.append(mjd_date_PTF_g[i])
#         p += 1
#         continue

# r_list = []
# r_unc_list = []
# r_av_mag = []
# r_av_uncs = []
# mjd_list_r = []
# mjd_date_r_epoch = []
# one_epoch_r = []
# one_epoch_r_unc = []
# m = 1
# p = 0
# for i in range(len(PTF_mag_r)):
#     if i == 0:
#         r_list.append(PTF_mag_r[i])
#         r_unc_list.append(PTF_unc_r[i])
#         mjd_list_r.append(mjd_date_PTF_r[i])
#         continue
#     elif i == len(PTF_mag_r) - 1:
#         r_av_mag.append(np.average(r_list))
#         r_av_uncs.append((1/len(r_unc_list))*np.sqrt(np.sum(np.square(r_unc_list))))
#         mjd_date_r_epoch.append(np.average(mjd_list_r))
#         if p == m:
#             one_epoch_r = r_list
#             one_epoch_r_unc = r_unc_list
#             one_epoch_r_mjd = mjd_list_r
#             p += 1
#     elif mjd_date_PTF_r[i] - mjd_date_PTF_r[i-1] < 100:
#         r_list.append(PTF_mag_r[i])
#         r_unc_list.append(PTF_unc_r[i])
#         mjd_list_r.append(mjd_date_PTF_r[i])
#         continue
#     else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
#         r_av_mag.append(np.average(r_list))
#         r_av_uncs.append((1/len(r_unc_list))*np.sqrt(np.sum(np.square(r_unc_list))))
#         mjd_date_r_epoch.append(np.average(mjd_list_r))
#         if p == m:
#             one_epoch_r = r_list
#             one_epoch_r_unc = r_unc_list
#             one_epoch_r_mjd = mjd_list_r
#             p += 1
#         r_list = []
#         r_unc_list = []
#         mjd_list_r = []
#         r_list.append(PTF_mag_r[i])
#         r_unc_list.append(PTF_unc_r[i])
#         mjd_list_r.append(mjd_date_PTF_r[i])
#         p += 1
#         continue

# # Changing mjd date to days since start:
# min_mjd = min([mjd_date_PTF_g[0], mjd_date_PTF_r[0], W1_av_mjd_date[0], W2_av_mjd_date[0]])
min_mjd = min([W1_av_mjd_date[0], W2_av_mjd_date[0]])
SDSS_mjd = SDSS_mjd - min_mjd
DESI_mjd = DESI_mjd - min_mjd
# mjd_date_g_epoch = [date - min_mjd for date in mjd_date_g_epoch]
# mjd_date_r_epoch = [date - min_mjd for date in mjd_date_r_epoch]
mjd_value = mjd_value - min_mjd
W1_av_mjd_date = [date - min_mjd for date in W1_av_mjd_date]
W2_av_mjd_date = [date - min_mjd for date in W2_av_mjd_date]

# print(f'Number of MIR W1 epochs = {len(W1_averages)}')
# print(f'Number of MIR W2 epochs = {len(W2_averages)}')

# # Plotting average raw flux vs mjd since first observation
# plt.figure(figsize=(12,7))
# # Flux
# plt.errorbar(W2_av_mjd_date, W2_averages, yerr=W2_av_uncs, fmt='o', color = 'blue', capsize=5, label = u'W2 (4.6 \u03bcm)')
# plt.errorbar(W1_av_mjd_date, W1_averages, yerr=W1_av_uncs, fmt='o', color = 'orange', capsize=5, label = u'W1 (3.4 \u03bcm)')
# # # Vertical line for SDSS & DESI dates:
# plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label = 'SDSS')
# plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label = 'DESI')
# # Labels and Titles
# plt.xlabel('Days since first observation')
# # Flux
# plt.ylabel('Flux / Units of digital numbers')
# plt.title(f'W1 & W2 Raw Flux vs Time ({object_name})')
# plt.legend(loc = 'best')
# plt.show()


W1_averages_flux = [flux(mag, W1_k, W1_wl) for mag in W1_averages]
W2_averages_flux = [flux(mag, W2_k, W2_wl) for mag in W2_averages]
W1_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W1_av_uncs, W1_averages_flux)] #See document in week 5 folder for conversion.
W2_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W2_av_uncs, W2_averages_flux)]
# g_averages_flux = [flux(mag, g_k, g_wl) for mag in g_av_mag]
# r_averages_flux = [flux(mag, r_k, r_wl) for mag in r_av_mag]
# g_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(g_av_uncs, g_averages_flux)]
# r_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(r_av_uncs, r_averages_flux)]

def find_closest_indices(x_vals, value):
    t = 0  
    if value <= x_vals[0]: #mjd is before first observation
        t += 1
        return 0, 0, t
    elif value >= x_vals[-1]: #mjd is after last observation
        t += 1
        return 0, 0, t
    for i in range(len(x_vals) - 1):
        if x_vals[i] <= value <= x_vals[i + 1]:
            before_index = i
            after_index = i + 1
            return before_index, after_index, t

before_SDSS_index_W1, after_SDSS_index_W1, q = find_closest_indices(W1_av_mjd_date, SDSS_mjd)
before_SDSS_index_W2, after_SDSS_index_W2, w = find_closest_indices(W2_av_mjd_date, SDSS_mjd)
before_DESI_index_W1, after_DESI_index_W1, e = find_closest_indices(W1_av_mjd_date, DESI_mjd)
before_DESI_index_W2, after_DESI_index_W2, r = find_closest_indices(W2_av_mjd_date, DESI_mjd)

if q == 0 and w == 0 and e == 0 and r == 0:
    
    #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to SDSS & DESI.
    W1_SDSS_interp = np.interp(SDSS_mjd, W1_av_mjd_date, W1_averages_flux)
    W2_SDSS_interp = np.interp(SDSS_mjd, W2_av_mjd_date, W2_averages_flux)
    W1_DESI_interp = np.interp(DESI_mjd, W1_av_mjd_date, W1_averages_flux)
    W2_DESI_interp = np.interp(DESI_mjd, W2_av_mjd_date, W2_averages_flux)

    #uncertainties in interpolated flux
    W1_SDSS_unc_interp = np.sqrt((((W1_av_mjd_date[after_SDSS_index_W1] - SDSS_mjd)/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[before_SDSS_index_W1])**2 + (((SDSS_mjd - W1_av_mjd_date[before_SDSS_index_W1])/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[after_SDSS_index_W1])**2)
    W2_SDSS_unc_interp = np.sqrt((((W2_av_mjd_date[after_SDSS_index_W2] - SDSS_mjd)/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[before_SDSS_index_W2])**2 + (((SDSS_mjd - W2_av_mjd_date[before_SDSS_index_W2])/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[after_SDSS_index_W2])**2)
    W1_DESI_unc_interp = np.sqrt((((W1_av_mjd_date[after_DESI_index_W1] - DESI_mjd)/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[before_DESI_index_W1])**2 + (((DESI_mjd - W1_av_mjd_date[before_DESI_index_W1])/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[after_DESI_index_W1])**2)
    W2_DESI_unc_interp = np.sqrt((((W2_av_mjd_date[after_DESI_index_W2] - DESI_mjd)/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[before_DESI_index_W2])**2 + (((DESI_mjd - W2_av_mjd_date[before_DESI_index_W2])/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[after_DESI_index_W2])**2)

    #uncertainty in absolute flux change
    W1_abs = abs(W1_SDSS_interp-W1_DESI_interp)
    W2_abs = abs(W2_SDSS_interp-W2_DESI_interp)
    W1_abs_unc = np.sqrt(W1_SDSS_unc_interp**2 + W1_DESI_unc_interp**2)
    W2_abs_unc = np.sqrt(W2_SDSS_unc_interp**2 + W2_DESI_unc_interp**2)

    #uncertainty in normalised flux change
    W1_av_unc = np.sqrt(sum(unc**2 for unc in W1_av_uncs_flux)) #uncertainty of the mean flux value
    W1_abs_norm = ((W1_abs)/(np.average(W1_averages_flux)))
    W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_av_unc)/(np.average(W1_averages_flux)))**2)
    W2_av_unc = np.sqrt(sum(unc**2 for unc in W2_av_uncs_flux)) #uncertainty of the mean flux value
    W2_abs_norm = ((W2_abs)/(np.average(W2_averages_flux)))
    W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_av_unc)/(np.average(W2_averages_flux)))**2)

    #uncertainty in z score
    W1_z_score_SDSS_DESI = (W1_SDSS_interp-W1_DESI_interp)/(W1_DESI_unc_interp)
    W1_z_score_SDSS_DESI_unc = W1_z_score_SDSS_DESI*((W1_abs_unc)/(W1_abs))
    W1_z_score_DESI_SDSS = (W1_DESI_interp-W1_SDSS_interp)/(W1_SDSS_unc_interp)
    W1_z_score_DESI_SDSS_unc = W1_z_score_DESI_SDSS*((W1_abs_unc)/(W1_abs))
    W2_z_score_SDSS_DESI = (W2_SDSS_interp-W2_DESI_interp)/(W2_DESI_unc_interp)
    W2_z_score_SDSS_DESI_unc = W2_z_score_SDSS_DESI*((W2_abs_unc)/(W2_abs))
    W2_z_score_DESI_SDSS = (W2_DESI_interp-W2_SDSS_interp)/(W2_SDSS_unc_interp)
    W2_z_score_DESI_SDSS_unc = W2_z_score_DESI_SDSS*((W2_abs_unc)/(W2_abs))


    #If uncertainty = nan; then z score = nan
    #If uncertainty = 0; then z score = inf
    print(f'W1 absolute flux change = {W1_abs} ± {W1_abs_unc}')
    print(f'W1 normalised absolute flux change = {W1_abs_norm} ± {W1_abs_norm_unc}')
    print(f'W1 z score - SDSS relative to DESI = {W1_z_score_SDSS_DESI} ± {W1_z_score_SDSS_DESI_unc}')
    print(f'W1 z score - DESI relative to SDSS = {W1_z_score_DESI_SDSS} ± {W1_z_score_DESI_SDSS_unc}')
    print(f'W2 absolute flux change = {W2_abs} ± {W2_abs_unc}')
    print(f'W2 normalised absolute flux change = {W2_abs_norm} ± {W2_abs_norm_unc}')
    print(f'W2 z score - SDSS relative to DESI = {W2_z_score_SDSS_DESI} ± {W2_z_score_SDSS_DESI_unc}')
    print(f'W2 z score - DESI relative to SDSS = {W2_z_score_DESI_SDSS} ± {W2_z_score_DESI_SDSS_unc}')

# # Plotting average W1 & W2 mags (or flux) vs days since first observation
# plt.figure(figsize=(12,7))
# # # Mag
# # plt.errorbar(W2_av_mjd_date, W2_averages, yerr=W2_av_uncs, fmt='o', color = 'blue', capsize=5, label = u'W2 (4.6 \u03bcm)')
# # plt.errorbar(W1_av_mjd_date, W1_averages, yerr=W1_av_uncs, fmt='o', color = 'orange', capsize=5, label = u'W1 (3.4 \u03bcm)') # fmt='o' makes the data points appear as circles.
# # Flux
# plt.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W2 (4.6 \u03bcm)')
# plt.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W1 (3.4 \u03bcm)')
# # # Vertical line for SDSS & DESI dates:
# plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label = 'SDSS')
# plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label = 'DESI')
# # Labels and Titles
# plt.xlabel('Days since first observation')
# # # Mag
# # plt.ylabel('Magnitude')
# # plt.title(f'W1 & W2 magnitude vs Time (SNR \u2265 {Min_SNR})')
# # Flux
# plt.ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# plt.title(f'W1 & W2 Flux vs Time ({object_name})')
# plt.legend(loc = 'best')
# plt.show()


# # Plotting W1 flux Extinction Corrected Vs Uncorrected
# inverse_W1_lamb = [1/3.4]*len(W1_averages_flux) #need units of inverse microns for extinguishing
# W1_corrected_flux = W1_averages_flux/ext_model.extinguish(inverse_W1_lamb, Ebv=mean_E_BV) #divide to remove the effect of dust

# plt.figure(figsize=(12,7))
# plt.scatter(W1_av_mjd_date, W1_corrected_flux, color = 'cyan', label = 'Corrected')
# plt.scatter(W1_av_mjd_date, W1_averages_flux, color = 'darkgreen', label = 'Uncorrected')
# plt.xlabel('Days since first observation')
# plt.ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# plt.title(f'W1 Flux ({object_name}) - Extinction Corrected vs Uncorrected')
# plt.legend(loc = 'best')
# plt.show()


# # Plotting colour (W1 mag[average] - W2 mag[average]):
# colour = [W1 - W2 for W1, W2 in zip(W1_averages, W2_averages)]
# colour_uncs = [np.sqrt((W1_unc_c)**2+(W2_unc_c)**2) for W1_unc_c, W2_unc_c in zip(W1_av_uncs, W2_av_uncs)]
# # Uncertainty propagation taken from Hughes & Hase; Z = A - B formula on back cover.

# plt.figure(figsize=(12,7))
# plt.errorbar(mjd_date_, colour, yerr=colour_uncs, fmt='o', color = 'red', capsize=5)
# #Labels and Titles
# plt.xlabel('Days since first observation')
# plt.ylabel('Colour')
# plt.title('Colour (W1 mag - W2 mag) vs Time')
# plt.show()


# # Specifically looking at a particular epoch:
# # Change 'm = _' in above code to change which epoch you look at. m = 0 represents epoch 1.
# plt.figure(figsize=(12,7))
# plt.errorbar(one_epoch_r_mjd, one_epoch_r, yerr=one_epoch_r_unc, fmt='o', color='red', capsize=5, label=u'PTF - r band')
# plt.title(f'r Band Measurements at Epoch {m+1} - {min([mjd_date_g_epoch[0], mjd_date_r_epoch[0]]):.0f} Days Since First WISE Observation', fontsize=16)
# plt.xlabel('mjd')
# plt.ylabel('Magnitude')
# plt.legend(loc='upper left')
# plt.show()

# (measurements are taken with a few days hence considered repeats)
# Create a figure with two subplots (1 row, 2 columns)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7), sharex=False)
# sharex = True explanation:
# Both subplots will have the same x-axis limits and tick labels.
# Any changes to the x-axis range (e.g., zooming or setting limits) in one subplot will automatically apply to the other subplot.

# data_point_W1 = list(range(1, len(one_epoch_W1) + 1))
# data_point_W2 = list(range(1, len(one_epoch_W2) + 1))

# # Plot in the first subplot (ax1)
# ax1.errorbar(one_epoch_r_mjd, one_epoch_r, yerr=one_epoch_r_unc, fmt='o', color='red', capsize=5, label=u'PTF - r band')
# ax1.set_title('r band')
# ax1.set_xlabel('mjd')
# ax1.set_ylabel('Magnitude')
# ax1.legend(loc='upper left')

# # Plot in the second subplot (ax2)
# ax2.errorbar(one_epoch_g_mjd, one_epoch_g, yerr=one_epoch_g_unc, fmt='o', color='green', capsize=5, label=u'PTF - g band')
# ax2.set_title('g band')
# ax2.set_xlabel('mjd')
# ax2.set_ylabel('Magnitude')
# ax2.legend(loc='upper left')

# fig.suptitle(f'g & r band Measurements at Epoch {m+1} - {min([mjd_date_g_epoch[0], mjd_date_r_epoch[0]]):.0f} Days Since First Observation', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
# plt.show()


# #Plotting a histogram of a single epoch
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))  # Creates a figure with 1 row and 2 columns

# bins_W1 = np.arange(min(one_epoch_W1), max(one_epoch_W1) + 0.05, 0.05)
# ax1.hist(one_epoch_W1, bins=bins_W1, color='orange', edgecolor='black')
# ax1.set_title('W1')
# ax1.set_xlabel('Magnitude')
# ax1.set_ylabel('Frequency')

# bins_W2 = np.arange(min(one_epoch_W2), max(one_epoch_W2) + 0.05, 0.05)
# ax2.hist(one_epoch_W2, bins=bins_W2, color='blue', edgecolor='black')
# ax2.set_title('W2')
# ax2.set_xlabel('Magnitude')
# ax2.set_ylabel('Frequency')

# plt.suptitle(f'W1 & W2 Magnitude Measurements at Epoch {m+1} - {mjd_value:.0f} Days Since First Observation', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
# plt.show()


# # Making a big figure with average mags & SDSS, DESI spectra added in
# fig = plt.figure(figsize=(12, 7))

# # common_ymin = -10
# # common_ymax = 20

# # Original big plot in the first row, spanning both columns (ax1)
# ax1 = fig.add_subplot(2, 1, 1)  # This will span the entire top row
# ax1.errorbar(mjd_date_, W1_averages, yerr=W1_av_uncs, fmt='o', color='orange', capsize=5, label=u'W1 (3.4 \u03bcm)')
# ax1.errorbar(mjd_date_, W2_averages, yerr=W2_av_uncs, fmt='o', color='blue', capsize=5, label=u'W2 (4.6 \u03bcm)')
# ax1.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS Observation')
# ax1.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI Observation')
# ax1.set_xlabel('Days since first observation')
# ax1.set_ylabel('Magnitude')
# ax1.set_title(f'W1 & W2 Magnitude vs Time (SNR \u2265 {Min_SNR})')
# ax1.legend(loc='upper left')

# # Create the two smaller plots side-by-side in the second row (ax2 and ax3)
# ax2 = fig.add_subplot(2, 2, 3)  # Left plot in the second row
# ax2.plot(sdss_lamb, sdss_flux, alpha = 0.2, color = 'forestgreen')
# ax2.plot(sdss_lamb, Gaus_smoothed_SDSS, color = 'forestgreen')
# if SDSS_min <= H_alpha <= SDSS_max:
#     ax2.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
# if SDSS_min <= H_beta <= SDSS_max:
#     ax2.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
# if SDSS_min <= Mg2 <= SDSS_max:
#     ax2.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
# if SDSS_min <= C3_ <= SDSS_max:
#     ax2.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
# if SDSS_min <= C4 <= SDSS_max:
#     ax2.axvline(C4, linewidth=2, color='violet', label = 'C IV')
# if SDSS_min <= _O3_ <= SDSS_max:
#     ax2.axvline(_O3_, linewidth=2, color='grey', label = '[O III]')
# if SDSS_min <= Ly_alpha <= SDSS_max:
#     ax2.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
# if SDSS_min <= Ly_beta <= SDSS_max:
#     ax2.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
# # ax2.set_xlabel('Wavelength / Å')
# ax2.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# # ax2.set_ylim(common_ymin, common_ymax)
# ax2.set_title('Gaussian Smoothed Plot of SDSS Spectrum')
# ax2.legend(loc='upper right')

# ax3 = fig.add_subplot(2, 2, 4)  # Right plot in the second row
# ax3.plot(desi_lamb, desi_flux, alpha = 0.2, color = 'midnightblue')
# ax3.plot(desi_lamb, Gaus_smoothed_DESI, color = 'midnightblue')
# if DESI_min <= H_alpha <= DESI_max:
#     ax3.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
# if DESI_min <= H_beta <= DESI_max:
#     ax3.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
# if DESI_min <= Mg2 <= DESI_max:
#     ax3.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
# if DESI_min <= C3_ <= DESI_max:
#     ax3.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
# if DESI_min <= C4 <= DESI_max:
#     ax3.axvline(C4, linewidth=2, color='violet', label = 'C IV')
# if DESI_min <= _O3_ <= DESI_max:
#     ax3.axvline(_O3_, linewidth=2, color='grey', label = '[O III]')
# if DESI_min <= Ly_alpha <= DESI_max:
#     ax3.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
# if DESI_min <= Ly_beta <= DESI_max:
#     ax3.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
# ax3.set_xlabel('Wavelength / Å')
# ax3.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# # ax3.set_ylim(common_ymin, common_ymax)
# ax3.set_title('Gaussian Smoothed Plot of DESI Spectrum')
# ax3.legend(loc='upper right')

# plt.show()


# Making a big figure with flux & SDSS, DESI spectra added in
fig = plt.figure(figsize=(12, 7)) # (width, height)
gs = GridSpec(5, 2, figure=fig)  # 5 rows, 2 columns

# common_ymin = -10
# common_ymax = 20

# Top plot spanning two columns and three rows (ax1)
ax1 = fig.add_subplot(gs[0:3, :])  # Rows 0 to 2, both columns
ax1.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color='blue', capsize=5, label=u'W2 (4.6 \u03bcm)')
ax1.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color='orange', capsize=5, label=u'W1 (3.4 \u03bcm)')
# ax1.errorbar(mjd_date_r_epoch, r_averages_flux, yerr=r_av_uncs_flux, fmt='o', color='red', capsize=5, label='r Band (616 nm)')
# ax1.errorbar(mjd_date_g_epoch, g_averages_flux, yerr=g_av_uncs_flux, fmt='o', color='green', capsize=5, label='g Band (467 nm)')
ax1.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS Observation')
ax1.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI Observation')
ax1.set_xlabel('Days since first observation')
ax1.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
ax1.set_title(f'Flux vs Time ({object_name})')
ax1.legend(loc='best')

# Bottom left plot spanning 2 rows and 1 column (ax2)
ax2 = fig.add_subplot(gs[3:, 0])  # Rows 3 to 4, first column
ax2.plot(sdss_lamb, sdss_flux, alpha=0.2, color='forestgreen')
ax2.plot(sdss_lamb, Gaus_smoothed_SDSS, color='forestgreen')
if SDSS_min <= H_alpha <= SDSS_max:
    ax2.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
if SDSS_min <= H_beta <= SDSS_max:
    ax2.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
if SDSS_min <= Mg2 <= SDSS_max:
    ax2.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
if SDSS_min <= C3_ <= SDSS_max:
    ax2.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
if SDSS_min <= C4 <= SDSS_max:
    ax2.axvline(C4, linewidth=2, color='violet', label = 'C IV')
# if SDSS_min <= _O3_ <= SDSS_max:
#     ax2.axvline(_O3_, linewidth=2, color='grey', label = '[O III]')
if SDSS_min <= Ly_alpha <= SDSS_max:
    ax2.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
if SDSS_min <= Ly_beta <= SDSS_max:
    ax2.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
ax2.set_xlabel('Wavelength / Å')
# ax2.set_ylim(common_ymin, common_ymax)
ax2.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
ax2.set_title('Gaussian Smoothed Plot of SDSS Spectrum')
ax2.legend(loc='best')

# Bottom right plot spanning 2 rows and 1 column (ax3)
ax3 = fig.add_subplot(gs[3:, 1])  # Rows 3 to 4, second column
ax3.plot(desi_lamb, desi_flux, alpha=0.2, color='midnightblue')
ax3.plot(desi_lamb, Gaus_smoothed_DESI, color='midnightblue')
if DESI_min <= H_alpha <= DESI_max:
    ax3.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
if DESI_min <= H_beta <= DESI_max:
    ax3.axvline(H_beta, linewidth=2, color='springgreen', label = u'H\u03B2')
if DESI_min <= Mg2 <= DESI_max:
    ax3.axvline(Mg2, linewidth=2, color='turquoise', label = 'Mg II')
if DESI_min <= C3_ <= DESI_max:
    ax3.axvline(C3_, linewidth=2, color='indigo', label = 'C III]')
if DESI_min <= C4 <= DESI_max:
    ax3.axvline(C4, linewidth=2, color='violet', label = 'C IV')
# if DESI_min <= _O3_ <= DESI_max:
#     ax3.axvline(_O3_, linewidth=2, color='grey', label = '[O III]')
if DESI_min <= Ly_alpha <= DESI_max:
    ax3.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
if DESI_min <= Ly_beta <= DESI_max:
    ax3.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
ax3.set_xlabel('Wavelength / Å')
# ax3.set_ylim(common_ymin, common_ymax)
ax3.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
ax3.set_title('Gaussian Smoothed Plot of DESI Spectrum')
ax3.legend(loc='best')

fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=1.25, wspace=0.2)
#top and bottom adjust the vertical space on the top and bottom of the figure.
#left and right adjust the horizontal space on the left and right sides.
#hspace and wspace adjust the spacing between rows and columns, respectively.

# fig.savefig(f'./CLAGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')

plt.show()