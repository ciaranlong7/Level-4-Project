import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.visualization import quantity_support
quantity_support()  # for getting units on the axes below

c = 299792458

#Open the SDSS file
with fits.open("spec-8521-58175-0279.fits") as hdul:
    subset = hdul[1]       

    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
    sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
    sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])

#Open the DESI file
DESI_spec = pd.read_csv('spectrum_desi_152517.57+401357.6.csv')
desi_lamb = DESI_spec.iloc[1:, 0]  # First column, skipping the first row (header)
desi_flux = DESI_spec.iloc[1:, 1]  # Second column, skipping the first row (header)

# Correcting for redshift.
z = 0.385
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
# plt.axvline(H_alpha, linewidth=2, color='goldenrod', label = 'H alpha')
# plt.axvline(H_beta, linewidth=2, color='green', label = 'H beta')
# plt.axvline(Mg2, linewidth=2, color='red', label = 'Mg ii')
# #Axis labels
# plt.xlabel('Wavelength / Å')
# plt.ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# #Two different titles (for Gaussian/Manual)
# plt.title('Gaussian Smoothed Plot of SDSS & DESI Spectra')
# # plt.title('Manually Smoothed Plot of SDSS & DESI Spectra')

# plt.legend(loc = 'upper right')
# plt.show()

#Plotting MIR data
def flux(mag, k): # k is the zero magnitude flux density. Taken from a data table on the search website
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys
W2_k = 171.787

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
colour = []
mjd_date_colour = []
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
if len(W1_mag) == len(W2_mag):
    i = 0
    j = 0
    k = 0
    x = 0 #skip flag for W2
    y = 0 #skip flag for W1
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
                    mjd_date_.append(W1_mag[i-k][1]) #Assumes that the mjd dates are so close that any difference between them is negligibile.
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
                    continue
            else: #This happens if the data goes BB, CB, BC, AA (and ph_qual lim set to >B)
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
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
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
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
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
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        continue
                else:
                    print('flag') #this path shouldn't ever be used, but need to think about it some more to double check
elif len(W1_mag) < len(W2_mag):
    i = 0
    j = 0
    k = 0
    x = 0 #skip flag for W2
    y = 0 #skip flag for W1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                    W1_list = []
                    W1_unc_list = []
                    W2_list = []
                    W2_unc_list = []
                    W1_list.append(W1_mag[i-k][0])
                    W1_unc_list.append(W1_mag[i-k][2])
                    W2_list.append(W2_mag[i-j][0])
                    W2_unc_list.append(W2_mag[i-j][2])
                    i += 1
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
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
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
                        W1_list = []
                        W1_unc_list = []
                        W2_list = []
                        W2_unc_list = []
                        W1_list.append(W1_mag[i-k][0])
                        W1_unc_list.append(W1_mag[i-k][2])
                        W2_list.append(W2_mag[i-j][0])
                        W2_unc_list.append(W2_mag[i-j][2])
                        i += 1
                        continue
                else:
                    print('flag') #this path shouldn't ever be used, but need to think about it some more to double check

# print(len(W1_mag))
# print(len(W2_mag))

# print(len(W1_averages))
# print(len(W2_averages))

#get SDSS & DESI mjd:
object_name = '152517.57+401357.6'
table_4_GUO = pd.read_csv('guo23_table4_clagn.csv')
object_data = table_4_GUO[table_4_GUO.iloc[:, 0] == object_name]
SDSS_mjd = object_data.iloc[0, 7]
DESI_mjd = object_data.iloc[0, 8]
SDSS_mjd = SDSS_mjd - mjd_date_[0]
DESI_mjd = DESI_mjd - mjd_date_[0]

#Changing mjd date to days since start:
mjd_date_ = [date - mjd_date_[0] for date in mjd_date_]

#Must zoom in on some of the epochs
# Also plot colour
# Also convert from mag to flux

plt.figure(figsize=(18,6))
#Averages:
plt.errorbar(mjd_date_, W1_averages, yerr=W1_av_uncs, fmt='o', color = 'orange', capsize=5, label = u'W1 (3.4 \u03bcm)') # fmt='o' makes the data points appear as circles.
plt.errorbar(mjd_date_, W2_averages, yerr=W2_av_uncs, fmt='o', color = 'blue', capsize=5, label = u'W2 (4.6 \u03bcm)')
#Vertical line for SDSS & DESI dates:
plt.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label = 'SDSS')
plt.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label = 'DESI')
#Labels and Titles
plt.xlabel('Days since first observation')
plt.ylabel('Magnitude')
plt.title('W1 & W2 magnitude vs time (ph_qual > B)')
plt.legend(loc = 'upper left')
plt.show()

# # Making a big figure with DESI & SDSS spectra added in
# fig = plt.figure(figsize=(18, 12))

# common_ymin = -10
# common_ymax = 20

# # Original big plot in the first row, spanning both columns (ax1)
# ax1 = fig.add_subplot(2, 1, 1)  # This will span the entire top row
# ax1.errorbar(mjd_date_, W1_averages, yerr=W1_av_uncs, fmt='o', color='orange', capsize=5, label=u'W1 (3.4 \u03bcm)')
# ax1.errorbar(mjd_date_, W2_averages, yerr=W2_av_uncs, fmt='o', color='blue', capsize=5, label=u'W2 (4.6 \u03bcm)')
# ax1.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS')
# ax1.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI')
# ax1.set_ylabel('Magnitude')
# ax1.set_title('W1 & W2 Magnitude vs Time (ph_qual > B)')
# ax1.legend(loc='upper left')

# # Create the two smaller plots side-by-side in the second row (ax2 and ax3)
# ax2 = fig.add_subplot(2, 2, 3)  # Left plot in the second row
# ax2.plot(sdss_lamb, sdss_flux, alpha = 0.2, color = 'forestgreen')
# ax2.plot(sdss_lamb, Gaus_smoothed_SDSS, color = 'forestgreen')
# ax2.axvline(H_alpha, linewidth=2, color='goldenrod', label = 'H alpha')
# ax2.axvline(H_beta, linewidth=2, color='green', label = 'H beta')
# ax2.axvline(Mg2, linewidth=2, color='red', label = 'Mg ii')
# ax2.set_xlabel('Wavelength / Å')
# ax2.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# # ax2.set_ylim(common_ymin, common_ymax)
# ax2.set_title('Gaussian Smoothed Plot of SDSS Spectra')
# ax2.legend(loc='upper left')

# ax3 = fig.add_subplot(2, 2, 4)  # Right plot in the second row
# ax3.plot(desi_lamb, desi_flux, alpha = 0.2, color = 'midnightblue')
# ax3.plot(desi_lamb, Gaus_smoothed_DESI, color = 'midnightblue')
# ax3.axvline(H_alpha, linewidth=2, color='goldenrod', label = 'H alpha')
# ax3.axvline(H_beta, linewidth=2, color='green', label = 'H beta')
# ax3.axvline(Mg2, linewidth=2, color='red', label = 'Mg ii')
# ax3.set_xlabel('Wavelength / Å')
# ax3.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
# # ax3.set_ylim(common_ymin, common_ymax)
# ax3.set_title('Gaussian Smoothed Plot of DESI Spectra')
# ax3.legend(loc='upper right')

# plt.show()