import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from astropy.io import fits
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.visualization import quantity_support
quantity_support()  # for getting units on the axes below

c = 299792458

#get SDSS & DESI filenames:
plate = 8521
fiberid = '0279'
object_name = '152517.57+401357.6'
table_4_GUO = pd.read_csv('guo23_table4_clagn.csv')
object_data = table_4_GUO[table_4_GUO.iloc[:, 0] == object_name]
SDSS_mjd = object_data.iloc[0, 7]
DESI_mjd = object_data.iloc[0, 8]
# SDSS_file = f'spec-{plate}-{SDSS_mjd:.0f}-{fiberid}.fits'
SDSS_file = 'spec-8521-58175-0279.fits'
DESI_file = f'spectrum_desi_{object_name}.csv'

#Open the SDSS file
SDSS_file_path = f'clagn_spectra/{SDSS_file}'
with fits.open(SDSS_file_path) as hdul:
    subset = hdul[1]

    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
    sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
    sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])

#Open the DESI file
DESI_file_path = f'clagn_spectra/{DESI_file}'
DESI_spec = pd.read_csv(DESI_file_path)
desi_lamb = DESI_spec.iloc[1:, 0]  # First column, skipping the first row (header)
desi_flux = DESI_spec.iloc[1:, 1]  # Second column, skipping the first row (header)

# Correcting for redshift.
z = object_data.iloc[0, 3]
sdss_lamb = sdss_lamb/(1+z)
desi_lamb = desi_lamb/(1+z)

#Calculate rolling average manually
def rolling_average(arr, window_size):
    
    averages = []
    for i in range(len(arr) - window_size + 1):
        avg = np.mean(arr[i:i + window_size])
        averages.append(avg)
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

# #Plot of SDSS & DESI Spectra
# plt.figure(figsize=(18,6))
# #Original unsmoothed spectrum
# plt.plot(sdss_lamb, sdss_flux, alpha = 0.2, color = 'orange')
# plt.plot(desi_lamb, desi_flux, alpha = 0.2, color = 'blue')
# #Gausian smoothing
# plt.plot(sdss_lamb, Gaus_smoothed_SDSS, color = 'orange', label = 'SDSS')
# plt.plot(desi_lamb, Gaus_smoothed_DESI, color = 'blue', label = 'DESI')
# #Manual smoothing
# # plt.plot(sdss_lamb, SDSS_rolling, color = 'orange', label = 'SDSS')
# # plt.plot(desi_lamb, DESI_rolling, color = 'blue', label = 'DESI')
# #Adding in positions of emission lines
# plt.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
# plt.axvline(H_beta, linewidth=2, color='green', label = u'H\u03B2')
# plt.axvline(Mg2, linewidth=2, color='red', label = 'Mg II')
# #Axis labels
# plt.xlabel('Wavelength / Å')
# plt.ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# #Two different titles (for Gaussian/Manual)
# plt.title('Gaussian Smoothed Plot of SDSS & DESI Spectra')
# # plt.title('Manually Smoothed Plot of SDSS & DESI Spectra')

# plt.legend(loc = 'upper right')
# plt.show()

#Plotting MIR data
#data must be filtered in terms order of mjd
MIR_data = pd.read_csv('Object_MIR_data.csv')

# Filter the DataFrame for rows where cc_flags is 0
filtered_rows = MIR_data[MIR_data.iloc[:, 15] == 0]

#Filtering for good SNR
# SNR > C
# filtered_rows_W1 = filtered_rows[filtered_rows.iloc[:, 16].isin(['AA', 'AB', 'AC', 'AU', 'AX', 'BA', 'BB', 'BC', 'BU', 'BX', 'CA', 'CB', 'CC', 'CU', 'CX'])]
# filtered_rows_W2 = filtered_rows[filtered_rows.iloc[:, 16].isin(['AA', 'BA', 'CA', 'UA', 'XA', 'AB', 'BB', 'CB', 'UB', 'XB', 'AC', 'BC', 'CC', 'UC', 'XC'])]
# SNR > B
filtered_rows_W1 = filtered_rows[filtered_rows.iloc[:, 16].isin(['AA', 'AB', 'AC', 'AU', 'AX', 'BA', 'BB', 'BC', 'BU', 'BX'])]
filtered_rows_W2 = filtered_rows[filtered_rows.iloc[:, 16].isin(['AA', 'BA', 'CA', 'UA', 'XA', 'AB', 'BB', 'CB', 'UB', 'XB'])]
# SNR > A
# filtered_rows_W1 = filtered_rows[filtered_rows.iloc[:, 16].isin(['AA', 'AB', 'AC', 'AU', 'AX'])]
# filtered_rows_W2 = filtered_rows[filtered_rows.iloc[:, 16].isin(['AA', 'BA', 'CA', 'UA', 'XA'])]

mjd_date_W1 = filtered_rows_W1.iloc[:, 18]
mjd_date_W1 = mjd_date_W1.tolist()
W1_mag = filtered_rows_W1.iloc[:, 5]
W1_unc = filtered_rows_W1.iloc[:, 6]
W1_mag = list(zip(W1_mag, mjd_date_W1, W1_unc))

mjd_date_W2 = filtered_rows_W2.iloc[:, 18]
mjd_date_W2 = mjd_date_W2.tolist()
W2_mag = filtered_rows_W2.iloc[:, 9]
W2_unc = filtered_rows_W2.iloc[:, 10]
W2_mag = list(zip(W2_mag, mjd_date_W2, W2_unc))

#The four W1_mag dps with ph_qual C are in rows, 29, 318, 386, 388

#Below code analyses MIR data.
#Only assumption required for code to work - there is never a situation where the data has only one data point for an epoch.
W1_list = []
W2_list = []
W1_unc_list = []
W2_unc_list = []
W1_averages= []
W2_averages = []
W1_av_uncs = []
W2_av_uncs = []
mjd_date_ = []
one_epoch_W1 = []
one_epoch_W1_unc = []
one_epoch_W2 = []
one_epoch_W2_unc = []
m = 4 # Change depending on which epoch you wish to look at. m = 0 represents epoch 1.
if len(W1_mag) == len(W2_mag):
    i = 0
    j = 0
    k = 0
    x = 0 #skip flag for W2
    y = 0 #skip flag for W1
    p = 0
    while i-k+1 < len(W1_mag):
        if W2_mag[i-j][1] != W1_mag[i-k][1]: #checking if mjd dates are the same.
            if W2_mag[i-j][1] > W1_mag[i-k][1]: #This means W2 list has skipped a reading (ie the skipped one had bad SNR)
                j += 1
                x += 1
                if i == 0:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    i += 1
                    continue
                elif W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100: #can guarantee no skip between W1_mag[i-k] & W_mag[i-k-1].
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list)))) #see derivation in week 5 folder.
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W1_mag[i-k][1]) #Assumes that the mjd dates are so close that any difference between them is negligible
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            elif W2_mag[i-j][1] < W1_mag[i-k][1]: #This means W1 list has skipped a reading (ie the skipped one had bad SNR)
                k += 1
                y += 1
                if i == 0:
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                elif W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W2_mag[i-j][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
        else: #mjd dates are the same
            if i == 0:
                W1_list.append(W1_mag[i-k][0])
                W1_unc_list.append(W1_mag[i-k][2])
                W2_list.append(W2_mag[i-j][0])
                W2_unc_list.append(W2_mag[i-j][2])
                i += 1
                continue
            elif x == 0: #checking no skip between two adjacent W2 data points; W2_mag[i-j][1] - W2_mag[i-j-1][1]
                y = 0
                if W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W2_mag[i-j][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            elif y == 0: #There was a skip between two adjacent W2 data points; W2_mag[i-j][1] - W2_mag[i-j-1][1]. Final check to see if skip between two adjacent W1 data points
                x = 0
                if W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W1_mag[i-k][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            else: #This happens if the data goes BB, CB, BC, AA (and ph_qual lim set to >= B)
                #All valid data points have already been stored. W1_mag[i-k] & W2_mag[i-j] corresponds to the AA data point in the example above
                x = 0
                y = 0
                if W1_mag[i-k][1] - W1_mag[i-k-1][1] < W2_mag[i-j][1] - W2_mag[i-j-1][1]: #checking if W1 or W2 had the previous valid reading (would be W1 in example above; Bc then AA)
                    if W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100:
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        continue
                    else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                        W1_averages.append(np.average(W1_list))
                        W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                        W2_averages.append(np.average(W2_list))
                        W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                        mjd_date_.append(W1_mag[i-k][1])
                        if p == m:
                            one_epoch_W1 = W1_list
                            one_epoch_W1_unc = W1_unc_list
                            one_epoch_W2 = W2_list
                            one_epoch_W2_unc = W2_unc_list
                            mjd_value = W1_mag[i-k][1]
                            p += 1
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        p += 1
                        continue
                elif W1_mag[i-k][1] - W1_mag[i-k-1][1] > W2_mag[i-j][1] - W2_mag[i-j-1][1]: #checking if W1 or W2 had the previous valid reading (would be W1 in example above; Bc then AA)
                    if W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        continue
                    else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                        W1_averages.append(np.average(W1_list))
                        W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                        W2_averages.append(np.average(W2_list))
                        W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                        mjd_date_.append(W2_mag[i-j][1])
                        if p == m:
                            one_epoch_W1 = W1_list
                            one_epoch_W1_unc = W1_unc_list
                            one_epoch_W2 = W2_list
                            one_epoch_W2_unc = W2_unc_list
                            mjd_value = W1_mag[i-k][1]
                            p += 1
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        p += 1
                        continue
                else:
                    print('flag') #this path shouldn't ever be used.
elif len(W1_mag) > len(W2_mag):
    i = 0
    j = 0
    k = 0
    x = 0 #skip flag for W2
    y = 0 #skip flag for W1
    p = 0 #for grabbing only one epoch's data
    while i-k+1 < len(W1_mag):
        if W2_mag[i-j][1] != W1_mag[i-k][1]: #checking if mjd dates are the same.
            if W2_mag[i-j][1] > W1_mag[i-k][1]: #This means W2 list has skipped a reading (ie the skipped one had bad SNR)
                j += 1
                x += 1
                if i == 0:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    i += 1
                    continue
                elif W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100: #can guarantee no skip between W1_mag[i-k] & W_mag[i-k-1].
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W1_mag[i-k][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            elif W2_mag[i-j][1] < W1_mag[i-k][1]: #This means W1 list has skipped a reading (ie the skipped one had bad SNR)
                k += 1
                y += 1
                if i == 0:
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                elif W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W2_mag[i-j][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
        else: #mjd dates are the same
            if i == 0:
                W1_list.append(W1_mag[i-k][0])
                W1_unc_list.append(W1_mag[i-k][2])
                W2_list.append(W2_mag[i-j][0])
                W2_unc_list.append(W2_mag[i-j][2])
                i += 1
                continue
            elif x == 0: #checking no skip between two adjacent W2 data points; W2_mag[i-j][1] - W2_mag[i-j-1][1]
                y = 0
                if W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W2_mag[i-j][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            elif y == 0: #There was a skip between two adjacent W2 data points; W2_mag[i-k][1] - W2_mag[i-k-1][1]. Final check to see if skip between two adjacent W1 data points
                x = 0
                if W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W1_mag[i-k][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            else: # There is a mistake in my resetting. Reset midway through an epoch sometimes.
                x = 0
                y = 0
                if W1_mag[i-k][1] - W1_mag[i-k-1][1] < W2_mag[i-j][1] - W2_mag[i-j-1][1]: #checking if W1 or W2 had the previous valid reading (would be W1 in example above; Bc then AA)
                    if W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100:
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        continue
                    else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                        W1_averages.append(np.average(W1_list))
                        W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                        W2_averages.append(np.average(W2_list))
                        W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                        mjd_date_.append(W1_mag[i-k][1])
                        if p == m:
                            one_epoch_W1 = W1_list
                            one_epoch_W1_unc = W1_unc_list
                            one_epoch_W2 = W2_list
                            one_epoch_W2_unc = W2_unc_list
                            mjd_value = W1_mag[i-k][1]
                            p += 1
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        p += 1
                        continue
                elif W1_mag[i-k][1] - W1_mag[i-k-1][1] > W2_mag[i-j][1] - W2_mag[i-j-1][1]: #checking if W1 or W2 had the previous valid reading (would be W1 in example above; Bc then AA)
                    if W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        continue
                    else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                        W1_averages.append(np.average(W1_list))
                        W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                        W2_averages.append(np.average(W2_list))
                        W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                        mjd_date_.append(W2_mag[i-j][1])
                        if p == m:
                            one_epoch_W1 = W1_list
                            one_epoch_W1_unc = W1_unc_list
                            one_epoch_W2 = W2_list
                            one_epoch_W2_unc = W2_unc_list
                            mjd_value = W1_mag[i-k][1]
                            p += 1
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        p += 1
                        continue
                else:
                    print('flag') #this path shouldn't ever be used.
elif len(W1_mag) < len(W2_mag):
    i = 0
    j = 0
    k = 0
    x = 0 #skip flag for W2
    y = 0 #skip flag for W1
    p = 0
    while i-j+1 < len(W2_mag):
        # print(f'j = {j}')
        # print(f'i = {i}')
        # print(f'k = {k}')
        if W2_mag[i-j][1] != W1_mag[i-k][1]: #checking if mjd dates are the same.
            if W2_mag[i-j][1] > W1_mag[i-k][1]: #This means W2 list has skipped a reading (ie the skipped one had bad SNR)
                j += 1
                x += 1
                if i == 0:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    i += 1
                    continue
                elif W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100: #can guarantee no skip between W1_mag[i-k] & W_mag[i-k-1].
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W1_mag[i-k][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            elif W2_mag[i-j][1] < W1_mag[i-k][1]: #This means W1 list has skipped a reading (ie the skipped one had bad SNR)
                k += 1
                y += 1
                if i == 0:
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                elif W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W2_mag[i-j][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
        else: #mjd dates are the same
            if i == 0:
                W1_list.append(W1_mag[i-k][0])
                W1_unc_list.append(W1_mag[i-k][2])
                W2_list.append(W2_mag[i-j][0])
                W2_unc_list.append(W2_mag[i-j][2])
                i += 1
                continue
            elif x == 0: #confirming no skip between two adjacent W2 data points; W2_mag[i-j][1] - W2_mag[i-j-1][1]
                y = 0
                if W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W2_mag[i-j][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            elif y == 0: #There was a skip between two adjacent W2 data points; W2_mag[i-j][1] - W2_mag[i-j-1][1]. Final check to see skip between two adjacent W1 data points
                x = 0
                if W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100:
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
                else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                    W1_averages.append(np.average(W1_list))
                    W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                    W2_averages.append(np.average(W2_list))
                    W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                    mjd_date_.append(W1_mag[i-k][1])
                    if p == m:
                        one_epoch_W1 = W1_list
                        one_epoch_W1_unc = W1_unc_list
                        one_epoch_W2 = W2_list
                        one_epoch_W2_unc = W2_unc_list
                        mjd_value = W1_mag[i-k][1]
                        p += 1
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    p += 1
                    continue
            else:
                x = 0
                y = 0
                if W1_mag[i-k][1] - W1_mag[i-k-1][1] < W2_mag[i-j][1] - W2_mag[i-j-1][1]: #checking if W1 or W2 had the previous valid reading (would be W1 in example above; Bc then AA)
                    if W1_mag[i-k][1] - W1_mag[i-k-1][1] < 100:
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        continue
                    else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                        W1_averages.append(np.average(W1_list))
                        W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                        W2_averages.append(np.average(W2_list))
                        W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                        mjd_date_.append(W1_mag[i-k][1])
                        if p == m:
                            one_epoch_W1 = W1_list
                            one_epoch_W1_unc = W1_unc_list
                            one_epoch_W2 = W2_list
                            one_epoch_W2_unc = W2_unc_list
                            mjd_value = W1_mag[i-k][1]
                            p += 1
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        p += 1
                        continue
                elif W1_mag[i-k][1] - W1_mag[i-k-1][1] > W2_mag[i-j][1] - W2_mag[i-j-1][1]: #checking if W1 or W2 had the previous valid reading (would be W1 in example above; Bc then AA)
                    if W2_mag[i-j][1] - W2_mag[i-j-1][1] < 100:
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        continue
                    else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                        W1_averages.append(np.average(W1_list))
                        W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
                        W2_averages.append(np.average(W2_list))
                        W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
                        mjd_date_.append(W2_mag[i-j][1])
                        if p == m:
                            one_epoch_W1 = W1_list
                            one_epoch_W1_unc = W1_unc_list
                            one_epoch_W2 = W2_list
                            one_epoch_W2_unc = W2_unc_list
                            mjd_value = W1_mag[i-k][1]
                            p += 1
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        p += 1
                        continue
                else:
                    print('flag') #this path shouldn't ever be used.

#Changing mjd date to days since start:
SDSS_mjd = SDSS_mjd - mjd_date_[0]
DESI_mjd = DESI_mjd - mjd_date_[0]
mjd_value = mjd_value - mjd_date_[0]
mjd_date_ = [date - mjd_date_[0] for date in mjd_date_]

print(f'Object Name = {object_name}')
print(f'W1 data points = {len(W1_mag)}')
print(f'W2 data points = {len(W2_mag)}')
print(f'Number of epochs = {len(W1_averages)}')

# Plotting ideas:
# Also plot SDSS & DESI colour (must know if colour is mag - mag or flux - flux first)
# Find a way to convert from SDSS & DESI flux to mag.
# Look into spectral fitting of DESI & SDSS spectra.

def flux(mag, k, wavel): # k is the zero magnitude flux density. Taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys
W2_k = 171.787
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4

W1_averages_flux = [flux(mag, W1_k, W1_wl) for mag in W1_averages]
W1_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W1_av_uncs, W1_averages_flux)] #See document in week 5 folder for conversion.
W2_averages_flux = [flux(mag, W2_k, W2_wl) for mag in W2_averages]
W2_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W2_av_uncs, W2_averages_flux)]

# Plotting average W1 & W2 mags (or flux) vs days since first observation
# plt.figure(figsize=(14,6))
# Mag
# plt.errorbar(mjd_date_, W1_averages, yerr=W1_av_uncs, fmt='o', color = 'orange', capsize=5, label = u'W1 (3.4 \u03bcm)') # fmt='o' makes the data points appear as circles.
# plt.errorbar(mjd_date_, W2_averages, yerr=W2_av_uncs, fmt='o', color = 'blue', capsize=5, label = u'W2 (4.6 \u03bcm)')
# Flux
# plt.errorbar(mjd_date_, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color = 'orange', capsize=5, label = u'W1 (3.4 \u03bcm)')
# plt.errorbar(mjd_date_, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color = 'blue', capsize=5, label = u'W2 (4.6 \u03bcm)')
#Vertical line for SDSS & DESI dates:
# plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label = 'SDSS')
# plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label = 'DESI')
#Labels and Titles
# plt.xlabel('Days since first observation')
# Mag
# plt.ylabel('Magnitude')
# plt.title('W1 & W2 magnitude vs Time (ph_qual \u2265 B)')
# Flux
# plt.ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# plt.title('W1 & W2 Flux vs Time (ph_qual \u2265 B)')
# plt.legend(loc = 'upper left')
# plt.show()


# # Plotting colour (W1 mag[average] - W2 mag[average]):
# colour = [W1 - W2 for W1, W2 in zip(W1_averages, W2_averages)]
# colour_uncs = [np.sqrt((W1_unc_c)**2+(W2_unc_c)**2) for W1_unc_c, W2_unc_c in zip(W1_av_uncs, W2_av_uncs)]
# # Uncertainty propagation taken from Hughes & Hase; Z = A - B formula on back cover.

# plt.figure(figsize=(14,6))
# plt.errorbar(mjd_date_, colour, yerr=colour_uncs, fmt='o', color = 'red', capsize=5)
# #Labels and Titles
# plt.xlabel('Days since first observation')
# plt.ylabel('Colour')
# plt.title('Colour (W1 mag - W2 mag) vs Time')
# plt.show()


# # Specifically looking at a particular epoch:
# # Change 'm = _' in above code to change which epoch you look at. m = 0 represents epoch 1.
# # If I zoom in on one group, I could then see if it is the uncertainties that are causing the ~0.5mag variation in the repeat measurements.
# # (measurements are taken with a few days hence considered repeats)
# # Create a figure with two subplots (1 row, 2 columns)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=False)
# # sharex = True explanation:
# # Both subplots will have the same x-axis limits and tick labels.
# # Any changes to the x-axis range (e.g., zooming or setting limits) in one subplot will automatically apply to the other subplot.

# data_point_W1 = list(range(1, len(one_epoch_W1) + 1))
# data_point_W2 = list(range(1, len(one_epoch_W2) + 1))

# # Plot in the first subplot (ax1)
# ax1.errorbar(data_point_W1, one_epoch_W1, yerr=one_epoch_W1_unc, fmt='o', color='orange', capsize=5, label=u'W1 (3.4 \u03bcm)')
# ax1.set_xlabel('Data Point')
# ax1.set_ylabel('Magnitude')
# ax1.legend(loc='upper left')

# # Plot in the second subplot (ax2)
# ax2.errorbar(data_point_W2, one_epoch_W2, yerr=one_epoch_W2_unc, fmt='o', color='blue', capsize=5, label=u'W2 (4.6 \u03bcm)')
# ax2.set_xlabel('Data Point')
# ax2.set_ylabel('Magnitude')
# ax2.legend(loc='upper left')

# fig.suptitle(f'W1 & W2 Magnitude Measurements at Epoch {m+1} - {mjd_value:.0f} Days Since First Observation', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
# plt.show()


# # Making a big figure with averages & SDSS, DESI spectra added in
# fig = plt.figure(figsize=(18, 12))

# common_ymin = -10
# common_ymax = 20

# # Original big plot in the first row, spanning both columns (ax1)
# ax1 = fig.add_subplot(2, 1, 1)  # This will span the entire top row
# ax1.errorbar(mjd_date_, W1_averages, yerr=W1_av_uncs, fmt='o', color='orange', capsize=5, label=u'W1 (3.4 \u03bcm)')
# ax1.errorbar(mjd_date_, W2_averages, yerr=W2_av_uncs, fmt='o', color='blue', capsize=5, label=u'W2 (4.6 \u03bcm)')
# ax1.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS Observation')
# ax1.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI Observation')
# ax1.set_xlabel('Days since first observation')
# ax1.set_ylabel('Magnitude')
# ax1.set_title('W1 & W2 Magnitude vs Time (ph_qual \u2265 B)')
# ax1.legend(loc='upper left')

# # Create the two smaller plots side-by-side in the second row (ax2 and ax3)
# ax2 = fig.add_subplot(2, 2, 3)  # Left plot in the second row
# ax2.plot(sdss_lamb, sdss_flux, alpha = 0.2, color = 'forestgreen')
# ax2.plot(sdss_lamb, Gaus_smoothed_SDSS, color = 'forestgreen')
# ax2.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
# ax2.axvline(H_beta, linewidth=2, color='green', label = u'H\u03B2')
# ax2.axvline(Mg2, linewidth=2, color='red', label = 'Mg II')
# ax2.set_xlabel('Wavelength / Å')
# ax2.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# # ax2.set_ylim(common_ymin, common_ymax)
# ax2.set_title('Gaussian Smoothed Plot of SDSS Spectrum')
# ax2.legend(loc='upper right')

# ax3 = fig.add_subplot(2, 2, 4)  # Right plot in the second row
# ax3.plot(desi_lamb, desi_flux, alpha = 0.2, color = 'midnightblue')
# ax3.plot(desi_lamb, Gaus_smoothed_DESI, color = 'midnightblue')
# ax3.axvline(H_alpha, linewidth=2, color='goldenrod', label = u'H\u03B1')
# ax3.axvline(H_beta, linewidth=2, color='green', label = u'H\u03B2')
# ax3.axvline(Mg2, linewidth=2, color='red', label = 'Mg II')
# ax3.set_xlabel('Wavelength / Å')
# ax3.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# # ax3.set_ylim(common_ymin, common_ymax)
# ax3.set_title('Gaussian Smoothed Plot of DESI Spectrum')
# ax3.legend(loc='upper right')

# plt.show()


# Making a big figure with flux & SDSS, DESI spectra added in
fig = plt.figure(figsize=(12, 7)) # (width, height)
gs = GridSpec(5, 2, figure=fig)  # 5 rows, 2 columns

# Top plot spanning two columns and three rows (ax1)
ax1 = fig.add_subplot(gs[0:3, :])  # Rows 0 to 2, both columns
ax1.errorbar(mjd_date_, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color='orange', capsize=5, label=u'W1 (3.4 \u03bcm)')
ax1.errorbar(mjd_date_, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color='blue', capsize=5, label=u'W2 (4.6 \u03bcm)')
ax1.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS Observation')
ax1.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI Observation')
ax1.set_xlabel('Days since first observation')
ax1.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
ax1.set_title(f'W1 & W2 Flux vs Time ({object_name})')
ax1.legend(loc='best')

# Bottom left plot spanning 2 rows and 1 column (ax2)
ax2 = fig.add_subplot(gs[3:, 0])  # Rows 3 to 4, first column
ax2.plot(sdss_lamb, sdss_flux, alpha=0.2, color='forestgreen')
ax2.plot(sdss_lamb, Gaus_smoothed_SDSS, color='forestgreen')
ax2.axvline(H_alpha, linewidth=2, color='goldenrod', label=u'H\u03B1')
ax2.axvline(H_beta, linewidth=2, color='green', label=u'H\u03B2')
ax2.axvline(Mg2, linewidth=2, color='red', label='Mg II')
ax2.set_xlabel('Wavelength / Å')
ax2.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
ax2.set_title('Gaussian Smoothed Plot of SDSS Spectrum')
ax2.legend(loc='upper right')

# Bottom right plot spanning 2 rows and 1 column (ax3)
ax3 = fig.add_subplot(gs[3:, 1])  # Rows 3 to 4, second column
ax3.plot(desi_lamb, desi_flux, alpha=0.2, color='midnightblue')
ax3.plot(desi_lamb, Gaus_smoothed_DESI, color='midnightblue')
ax3.axvline(H_alpha, linewidth=2, color='goldenrod', label=u'H\u03B1')
ax3.axvline(H_beta, linewidth=2, color='green', label=u'H\u03B2')
ax3.axvline(Mg2, linewidth=2, color='red', label='Mg II')
ax3.set_xlabel('Wavelength / Å')
ax3.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
ax3.set_title('Gaussian Smoothed Plot of DESI Spectrum')
ax3.legend(loc='upper right')

fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=1.25, wspace=0.2)
#top and bottom adjust the vertical space on the top and bottom of the figure.
#left and right adjust the horizontal space on the left and right sides.
#hspace and wspace adjust the spacing between rows and columns, respectively.
plt.show()