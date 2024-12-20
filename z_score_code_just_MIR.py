import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa

c = 299792458

parent_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")

# #When changing object names list from CLAGN to AGN - I must change the files I am saving to at the bottom as well.
object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

# #When changing object names list from CLAGN to AGN - I must change the files I am saving to at the bottom as well.
# object_names = parent_sample.iloc[:, 3].sample(n=250, random_state=42) #randomly selecting 250 object names from clean parent sample

def flux(mag, k, wavel): # k is the zero magnitude flux density. For W1 & W2, taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
    k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
    return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys. This means that mag 0 = 309.540 Janskys at the W1 wl.
W2_k = 171.787
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4

object_names_list = [] #Keeps track of objects that met MIR data requirements to take z score & absolute change
SDSS_redshifts = []
DESI_redshifts = []

# z_score & absolute change lists
W1_max = []
W1_max_unc = []
W1_min = []
W1_min_unc = []
W1_abs_change = []
W1_abs_change_unc = []
W1_abs_change_norm = []
W1_abs_change_norm_unc = []
W1_gap = []

W2_max = []
W2_max_unc = []
W2_min = []
W2_min_unc = []
W2_abs_change = []
W2_abs_change_unc = []
W2_abs_change_norm = []
W2_abs_change_norm_unc = []
W2_gap = []

mean_zscore = []
mean_zscore_unc = []
mean_norm_flux_change = []
mean_norm_flux_change_unc = []

Min_SNR = 3 #Options are 10, 3, or 2. #A (SNR>10), B (3<SNR<10) or C (2<SNR<3)
if Min_SNR == 10: #Select Min_SNR on line above.
    MIR_SNR = 'A'
elif Min_SNR == 3:
    MIR_SNR = 'B'
elif Min_SNR == 2:
    MIR_SNR = 'C'
else:
    print('select a valid min SNR - 10, 3 or 2.')

g = 0
for object_name in object_names:
    print(g)
    print(object_name)
    g += 1
    # # For AGN:
    # object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
    # SDSS_RA = object_data.iloc[0, 0]
    # SDSS_DEC = object_data.iloc[0, 1]
    # SDSS_z = object_data.iloc[0, 2]
    # DESI_z = object_data.iloc[0, 9]

    #For CLAGN:
    object_data = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    SDSS_RA = object_data.iloc[0, 1]
    SDSS_DEC = object_data.iloc[0, 2]
    SDSS_z = object_data.iloc[0, 3]
    DESI_z = object_data.iloc[0, 3]

    # Automatically querying catalogues
    coord = SkyCoord(SDSS_RA, SDSS_DEC, unit='deg', frame='icrs') #This works.
    WISE_query = Irsa.query_region(coordinates=coord, catalog="allwise_p3as_mep", spatial="Cone", radius=2 * u.arcsec)
    NEOWISE_query = Irsa.query_region(coordinates=coord, catalog="neowiser_p1bs_psd", spatial="Cone", radius=2 * u.arcsec)
    WISE_data = WISE_query.to_pandas()
    NEO_data = NEOWISE_query.to_pandas()

    WISE_data = WISE_data.sort_values(by=WISE_data.columns[10]) #sort in ascending mjd
    NEO_data = NEO_data.sort_values(by=NEO_data.columns[42]) #sort in ascending mjd

    WISE_data.iloc[:, 6] = pd.to_numeric(WISE_data.iloc[:, 6], errors='coerce')
    filtered_WISE_rows = WISE_data[(WISE_data.iloc[:, 6] == 0) & (WISE_data.iloc[:, 39] == 1) & (WISE_data.iloc[:, 41] == '0000') & (WISE_data.iloc[:, 40] > 5)]
    #filtering for cc_flags == 0 in all bands, qi_fact == 1, no moon masking flag & separation of the WISE instrument to the SAA > 5 degrees. Unlike with Neowise, there is no individual column for cc_flags in each band

    filtered_NEO_rows = NEO_data[(NEO_data.iloc[:, 37] == 1) & (NEO_data.iloc[:, 38] > 5)] #checking for rows where qi_fact == 1 & separation of the WISE instrument to the South Atlantic Anomaly is > 5 degrees
    #"Single-exposure source database entries having qual_frame=0 should be used with extreme caution" - from the column descriptions.
    # The qi_fact column seems to be equal to qual_frame/10.

    #Filtering for good SNR, no cc_flags & no moon scattering flux
    if MIR_SNR == 'C':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX', 'BA', 'BB', 'BC', 'BU', 'BX', 'CA', 'CB', 'CC', 'CU', 'CX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA', 'AB', 'BB', 'CB', 'UB', 'XB', 'AC', 'BC', 'CC', 'UC', 'XC'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]
    elif MIR_SNR == 'B':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX', 'BA', 'BB', 'BC', 'BU', 'BX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA', 'AB', 'BB', 'CB', 'UB', 'XB'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]
    elif MIR_SNR == 'A':
        filtered_NEO_rows_W1 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'AB', 'AC', 'AU', 'AX'])) & (filtered_NEO_rows.iloc[:, 44] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '01']))]
        filtered_NEO_rows_W2 = filtered_NEO_rows[(filtered_NEO_rows.iloc[:, 34].isin(['AA', 'BA', 'CA', 'UA', 'XA'])) & (filtered_NEO_rows.iloc[:, 46] == '') & (filtered_NEO_rows.iloc[:, 39].isin(['00', '10']))]

    mjd_date_W1 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W1.iloc[:, 42].tolist()
    W1_mag = filtered_WISE_rows.iloc[:, 11].tolist() + filtered_NEO_rows_W1.iloc[:, 18].tolist()
    W1_unc = filtered_WISE_rows.iloc[:, 12].tolist() + filtered_NEO_rows_W1.iloc[:, 19].tolist()
    W1_mag = list(zip(W1_mag, mjd_date_W1, W1_unc))
    W1_mag = [tup for tup in W1_mag if not np.isnan(tup[0])] #removing instances where the mag value is NaN

    mjd_date_W2 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W2.iloc[:, 42].tolist()
    W2_mag = filtered_WISE_rows.iloc[:, 14].tolist() + filtered_NEO_rows_W2.iloc[:, 22].tolist()
    W2_unc = filtered_WISE_rows.iloc[:, 15].tolist() + filtered_NEO_rows_W2.iloc[:, 23].tolist()
    W2_mag = list(zip(W2_mag, mjd_date_W2, W2_unc))
    W2_mag = [tup for tup in W2_mag if not np.isnan(tup[0])]

    if len(W1_mag) < 2 and len(W2_mag) < 2: #checking if there is enough data
        print('No W1 & W2 data')
        continue

    #Below code sorts MIR data.
    #Two assumptions required for code to work:
    #1. The data is sorted in order of oldest mjd to most recent.
    #2. There are 2 or more data points.

    # W1 data first
    if len(W1_mag) > 1:
        W1_list = []
        W1_unc_list = []
        W1_mjds = []
        W1_data = []
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
                # W1_data.append( ( np.median(W1_list), np.median(W1_mjds), (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))) ) )
                if len(W1_list) > 1:
                    W1_data.append( ( np.median(W1_list), np.median(W1_mjds), median_abs_deviation(W1_list) ) )
                else:
                    W1_data.append( ( np.median(W1_list), np.median(W1_mjds), W1_unc_list[0] ) )
                continue
            elif W1_mag[i][1] - W1_mag[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
                W1_list.append(W1_mag[i][0])
                W1_mjds.append(W1_mag[i][1])
                W1_unc_list.append(W1_mag[i][2])
                continue
            else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                # W1_data.append( ( np.median(W1_list), np.median(W1_mjds), (1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))) ) )
                if len(W1_list) > 1:
                    W1_data.append( ( np.median(W1_list), np.median(W1_mjds), median_abs_deviation(W1_list) ) )
                else:
                    W1_data.append( ( np.median(W1_list), np.median(W1_mjds), W1_unc_list[0] ) )
                W1_list = []
                W1_mjds = []
                W1_unc_list = []
                W1_list.append(W1_mag[i][0])
                W1_mjds.append(W1_mag[i][1])
                W1_unc_list.append(W1_mag[i][2])
                continue
        #out of for loop now
    else:
        W1_data = [ (0,0,0) ]

    # W2 data second
    if len(W2_mag) > 1:
        W2_list = []
        W2_unc_list = []
        W2_mjds = []
        W2_data = []
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
                # W2_data.append( ( np.median(W2_list), np.median(W2_mjds), (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))) ) )
                if len(W2_list) > 1:
                    W2_data.append( ( np.median(W2_list), np.median(W2_mjds), median_abs_deviation(W2_list) ) )
                else:
                    W2_data.append( ( np.median(W2_list), np.median(W2_mjds), W2_unc_list[0] ) )
                continue
            elif W2_mag[i][1] - W2_mag[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
                W2_list.append(W2_mag[i][0])
                W2_mjds.append(W2_mag[i][1])
                W2_unc_list.append(W2_mag[i][2])
                continue
            else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
                # W2_data.append( ( np.median(W2_list), np.median(W2_mjds), (1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))) ) )
                if len(W2_list) > 1:
                    W2_data.append( ( np.median(W2_list), np.median(W2_mjds), median_abs_deviation(W2_list) ) )
                else:
                    W2_data.append( ( np.median(W2_list), np.median(W2_mjds), W2_unc_list[0] ) )
                W2_list = []
                W2_mjds = []
                W2_unc_list = []
                W2_list.append(W2_mag[i][0])
                W2_mjds.append(W2_mag[i][1])
                W2_unc_list.append(W2_mag[i][2])
                continue
    else:
        W2_data = [ (0,0,0) ]

    #want a minimum of 8 (out of ~25 possible) epochs to conduct analysis on.
    if len(W1_data) > 8:
        m = 0
    else:
        m = 1
    if len(W2_data) > 8:
        n = 0
    else:
        n = 1
    if m == 1 and n == 1:
        print('Bad W1 & W2 data')
        continue

    if m == 0: #Good W1 if true
        if n == 0: #Good W2 if true
            #Good W1 & W2
            object_names_list.append(object_name)
            SDSS_redshifts.append(SDSS_z)
            DESI_redshifts.append(DESI_z)

            min_mjd = min([W1_data[0][1], W2_data[0][1]])

            W1_av_mjd_date = [tup[1] - min_mjd for tup in W1_data]
            W2_av_mjd_date = [tup[1] - min_mjd for tup in W2_data]

            W1_averages_flux = [flux(tup[0], W1_k, W1_wl) for tup in W1_data]
            W1_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W1_data, W1_averages_flux)] #See document in week 5 folder for conversion.
            W2_averages_flux = [flux(tup[0], W2_k, W2_wl) for tup in W2_data]
            W2_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W2_data, W2_averages_flux)]

            W1_second_largest = sorted(W1_averages_flux, reverse=True)[1] #take second smallest and second largest to avoid sputious measurements. 
            W1_second_largest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_second_largest)] #NOT the 2nd largest unc. This is the unc in the second largest flux value
            W1_second_smallest = sorted(W1_averages_flux)[1]
            W1_second_smallest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_second_smallest)]

            #uncertainty in absolute flux change
            W1_abs = abs(W1_second_largest-W1_second_smallest)
            W1_abs_unc = np.sqrt(W1_second_largest_unc**2 + W1_second_smallest_unc**2)

            #uncertainty in normalised flux change
            W1_abs_norm = ((W1_abs)/(W1_second_smallest))
            W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_second_smallest_unc)/(W1_second_smallest))**2)

            #uncertainty in z score
            W1_z_score_max = (W1_second_largest-W1_second_smallest)/(W1_second_largest_unc)
            W1_z_score_max_unc = abs(W1_z_score_max*((W1_abs_unc)/(W1_abs)))
            W1_z_score_min = (W1_second_smallest-W1_second_largest)/(W1_second_smallest_unc)
            W1_z_score_min_unc = abs(W1_z_score_min*((W1_abs_unc)/(W1_abs)))

            W1_max.append(W1_z_score_max)
            W1_max_unc.append(W1_z_score_max_unc)
            W1_min.append(W1_z_score_min)
            W1_min_unc.append(W1_z_score_min_unc)
            W1_abs_change.append(W1_abs)
            W1_abs_change_unc.append(W1_abs_unc)
            W1_abs_change_norm.append(W1_abs_norm)
            W1_abs_change_norm_unc.append(W1_abs_norm_unc)

            W1_gap.append(abs(W1_av_mjd_date[W1_averages_flux.index(W1_second_largest)] - W1_av_mjd_date[W1_averages_flux.index(W1_second_smallest)]))

            W2_second_largest = sorted(W2_averages_flux, reverse=True)[1] #take second smallest and second largest to avoid sputious measurements. 
            W2_second_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_second_largest)] #NOT the 2nd largest unc. This is the unc in the second largest flux value
            W2_second_smallest = sorted(W2_averages_flux)[1]
            W2_second_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_second_smallest)]

            W2_abs = abs(W2_second_largest-W2_second_smallest)
            W2_abs_unc = np.sqrt(W2_second_largest_unc**2 + W2_second_smallest_unc**2)

            W2_abs_norm = ((W2_abs)/(W2_second_smallest))
            W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_second_smallest_unc)/(W2_second_smallest))**2)

            W2_z_score_max = (W2_second_largest-W2_second_smallest)/(W2_second_largest_unc)
            W2_z_score_max_unc = abs(W2_z_score_max*((W2_abs_unc)/(W2_abs)))
            W2_z_score_min = (W2_second_smallest-W2_second_largest)/(W2_second_smallest_unc)
            W2_z_score_min_unc = abs(W2_z_score_min*((W2_abs_unc)/(W2_abs)))

            W2_max.append(W2_z_score_max)
            W2_max_unc.append(W2_z_score_max_unc)
            W2_min.append(W2_z_score_min)
            W2_min_unc.append(W2_z_score_min_unc)
            W2_abs_change.append(W2_abs)
            W2_abs_change_unc.append(W2_abs_unc)
            W2_abs_change_norm.append(W2_abs_norm)
            W2_abs_change_norm_unc.append(W2_abs_norm_unc)

            W2_gap.append(abs(W2_av_mjd_date[W2_averages_flux.index(W2_second_largest)] - W2_av_mjd_date[W2_averages_flux.index(W2_second_smallest)]))

            zscores = np.sort([abs(W1_z_score_max), abs(W1_z_score_min), abs(W2_z_score_max), abs(W2_z_score_min)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_max_unc, W1_z_score_min_unc, W2_z_score_max_unc, W2_z_score_min_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/4)*np.sqrt(sum(unc**2 for unc in zscore_uncs)))
            
            norm_f_ch = np.sort([W1_abs_norm, W2_abs_norm])
            norm_f_ch_unc = np.sort([W1_abs_norm_unc, W2_abs_norm_unc])
            if np.isnan(norm_f_ch[0]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be nan
                mean_norm_flux_change_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_norm_flux_change_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch))
                mean_norm_flux_change_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))
            
        else: 
            #good W1, bad W2
            object_names_list.append(object_name)
            SDSS_redshifts.append(SDSS_z)
            DESI_redshifts.append(DESI_z)

            min_mjd = W1_data[0][1]

            W1_av_mjd_date = [tup[1] - min_mjd for tup in W1_data]

            W1_averages_flux = [flux(tup[0], W1_k, W1_wl) for tup in W1_data]
            W1_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W1_data, W1_averages_flux)]

            W1_second_largest = sorted(W1_averages_flux, reverse=True)[1]
            W1_second_largest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_second_largest)]
            W1_second_smallest = sorted(W1_averages_flux)[1]
            W1_second_smallest_unc = W1_av_uncs_flux[W1_averages_flux.index(W1_second_smallest)]

            W1_abs = abs(W1_second_largest-W1_second_smallest)
            W1_abs_unc = np.sqrt(W1_second_largest_unc**2 + W1_second_smallest_unc**2)

            W1_abs_norm = ((W1_abs)/(W1_second_smallest))
            W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_second_smallest_unc)/(W1_second_smallest))**2)

            W1_z_score_max = (W1_second_largest-W1_second_smallest)/(W1_second_largest_unc)
            W1_z_score_max_unc = abs(W1_z_score_max*((W1_abs_unc)/(W1_abs)))
            W1_z_score_min = (W1_second_smallest-W1_second_largest)/(W1_second_smallest_unc)
            W1_z_score_min_unc = abs(W1_z_score_min*((W1_abs_unc)/(W1_abs)))

            W1_max.append(W1_z_score_max)
            W1_max_unc.append(W1_z_score_max_unc)
            W1_min.append(W1_z_score_min)
            W1_min_unc.append(W1_z_score_min_unc)
            W1_abs_change.append(W1_abs)
            W1_abs_change_unc.append(W1_abs_unc)
            W1_abs_change_norm.append(W1_abs_norm)
            W1_abs_change_norm_unc.append(W1_abs_norm_unc)

            W1_gap.append(abs(W1_av_mjd_date[W1_averages_flux.index(W1_second_largest)] - W1_av_mjd_date[W1_averages_flux.index(W1_second_smallest)]))

            W2_z_score_max = np.nan
            W2_z_score_max_unc = np.nan
            W2_z_score_min = np.nan
            W2_z_score_min_unc = np.nan

            W2_max.append(W2_z_score_max)
            W2_max_unc.append(W2_z_score_max_unc)
            W2_min.append(W2_z_score_min)
            W2_min_unc.append(W2_z_score_min_unc)
            W2_abs_change.append(np.nan)
            W2_abs_change_unc.append(np.nan)
            W2_abs_change_norm.append(np.nan)
            W2_abs_change_norm_unc.append(np.nan)

            W2_gap.append(np.nan)

            zscores = np.sort([abs(W1_z_score_max), abs(W1_z_score_min), abs(W2_z_score_max), abs(W2_z_score_min)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_max_unc, W1_z_score_min_unc, W2_z_score_max_unc, W2_z_score_min_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/4)*np.sqrt(sum(unc**2 for unc in zscore_uncs)))

            norm_f_ch = np.sort([W1_abs_norm, W2_abs_norm])
            norm_f_ch_unc = np.sort([W1_abs_norm_unc, W2_abs_norm_unc])
            if np.isnan(norm_f_ch[0]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be nan
                mean_norm_flux_change_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_norm_flux_change_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch))
                mean_norm_flux_change_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))

    else: #Bad W1
        if n == 0: #Good W2 if true
            #Bad W1, good W2
            object_names_list.append(object_name)
            SDSS_redshifts.append(SDSS_z)
            DESI_redshifts.append(DESI_z)

            min_mjd = W2_data[0][1]

            W2_av_mjd_date = [tup[1] - min_mjd for tup in W2_data]

            W2_av_uncs_flux = [((tup[2]*np.log(10))/(2.5))*flux for tup, flux in zip(W2_data, W2_averages_flux)]
            W2_averages_flux = [flux(tup[0], W2_k, W2_wl) for tup in W2_data]

            W1_z_score_max = np.nan
            W1_z_score_max_unc = np.nan
            W1_z_score_min = np.nan
            W1_z_score_min_unc = np.nan

            W1_max.append(W1_z_score_max)
            W1_max_unc.append(W1_z_score_max_unc)
            W1_min.append(W1_z_score_min)
            W1_min_unc.append(W1_z_score_min_unc)
            W1_abs_change.append(np.nan)
            W1_abs_change_unc.append(np.nan)
            W1_abs_change_norm.append(np.nan)
            W1_abs_change_norm_unc.append(np.nan)

            W1_gap.append(np.nan)

            W2_second_largest = sorted(W2_averages_flux, reverse=True)[1]
            W2_second_largest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_second_largest)]
            W2_second_smallest = sorted(W2_averages_flux)[1]
            W2_second_smallest_unc = W2_av_uncs_flux[W2_averages_flux.index(W2_second_smallest)]

            W2_abs = abs(W2_second_largest-W2_second_smallest)
            W2_abs_unc = np.sqrt(W2_second_largest_unc**2 + W2_second_smallest_unc**2)

            W2_abs_norm = ((W2_abs)/(W2_second_smallest))
            W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_second_smallest_unc)/(W2_second_smallest))**2)

            W2_z_score_max = (W2_second_largest-W2_second_smallest)/(W2_second_largest_unc)
            W2_z_score_max_unc = abs(W2_z_score_max*((W2_abs_unc)/(W2_abs)))
            W2_z_score_min = (W2_second_smallest-W2_second_largest)/(W2_second_smallest_unc)
            W2_z_score_min_unc = abs(W2_z_score_min*((W2_abs_unc)/(W2_abs)))

            W2_max.append(W2_z_score_max)
            W2_max_unc.append(W2_z_score_max_unc)
            W2_min.append(W2_z_score_min)
            W2_min_unc.append(W2_z_score_min_unc)
            W2_abs_change.append(W2_abs)
            W2_abs_change_unc.append(W2_abs_unc)
            W2_abs_change_norm.append(W2_abs_norm)
            W2_abs_change_norm_unc.append(W2_abs_norm_unc)

            W2_gap.append(abs(W2_av_mjd_date[W2_averages_flux.index(W2_second_largest)] - W2_av_mjd_date[W2_averages_flux.index(W2_second_smallest)]))

            zscores = np.sort([abs(W1_z_score_max), abs(W1_z_score_min), abs(W2_z_score_max), abs(W2_z_score_min)]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_max_unc, W1_z_score_min_unc, W2_z_score_max_unc, W2_z_score_min_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(zscores)) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(zscores))
                mean_zscore_unc.append((1/4)*np.sqrt(sum(unc**2 for unc in zscore_uncs)))
            
            norm_f_ch = np.sort([W1_abs_norm, W2_abs_norm])
            norm_f_ch_unc = np.sort([W1_abs_norm_unc, W2_abs_norm_unc])
            if np.isnan(norm_f_ch[0]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be nan
                mean_norm_flux_change_unc.append(np.nan)
            elif np.isnan(norm_f_ch[1]) == True:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch)) #will be norm_f_ch[0]
                mean_norm_flux_change_unc.append(abs(norm_f_ch_unc[0]))
            else:
                mean_norm_flux_change.append(np.nanmean(norm_f_ch))
                mean_norm_flux_change_unc.append((1/2)*np.sqrt(sum(unc**2 for unc in norm_f_ch_unc)))
  
        else:
            #bad W1, bad W2. Should've already 'continued' above to save time.
            print('Bad W1 & W2 data')
            continue

#for loop now ended
quantifying_change_data = {
    "Object": object_names_list, #0

    "W1 Z Score using Max Unc": W1_max, #1
    "Uncertainty in W1 Z Score using Max Unc": W1_max_unc, #2
    "W1 Z Score using Min Unc": W1_min, #3
    "Uncertainty in W1 Z Score using Min Unc": W1_min_unc, #4
    "W1 Flux Change": W1_abs_change, #5
    "W1 Flux Change Unc": W1_abs_change_unc, #6
    "W1 Normalised Flux Change": W1_abs_change_norm, #7
    "W1 Normalised Flux Change Unc": W1_abs_change_norm_unc, #8

    "W2 Z Score using Max Unc": W2_max, #9
    "Uncertainty in W2 Z Score using Max Unc": W2_max_unc, #10
    "W2 Z Score using Min Unc": W2_min, #11
    "Uncertainty in W2 Z Score using Min Unc": W2_min_unc, #12
    "W2 Flux Change": W2_abs_change, #13
    "W2 Flux Change Unc": W2_abs_change_unc, #14
    "W2 Normalised Flux Change": W2_abs_change_norm, #15
    "W2 Normalised Flux Change Unc": W2_abs_change_norm_unc, #16

    "Mean Z Score": mean_zscore, #17
    "Mean Z Score Unc": mean_zscore_unc, #18
    "Mean Normalised Flux Change": mean_norm_flux_change, #19
    "Mean Normalised Flux Change Unc": mean_norm_flux_change_unc, #20

    "W1 Gap": W1_gap, #21
    "W2 Gap": W2_gap, #22
    "SDSS Redshift": SDSS_redshifts, #23
    "DESI Redshift": DESI_redshifts, #24
}

# Convert the data into a DataFrame
df = pd.DataFrame(quantifying_change_data)

#Creating a csv file of my data
df.to_csv("CLAGN_Quantifying_Change_just_MIR_2nd_biggest_smallest.csv", index=False)
# df.to_csv("AGN_Quantifying_Change_just_MIR_2nd_biggest_smallest.csv", index=False)