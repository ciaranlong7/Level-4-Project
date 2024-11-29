import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
from astropy.io.fits.hdu.hdulist import HDUList
from astroquery.sdss import SDSS
from sparcl.client import SparclClient
from dl import queryClient as qc
import sfdmap
from dust_extinction.parameter_averages import G23

c = 299792458

# #When changing object names list from CLAGN to AGN - I must change the files I am saving to at the bottom as well.
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

# #random list of object names taken from parent catalogue
# object_names = ['085817.56+322349.7', '130115.40+252726.3', '101834.35+331258.9', '150210.72+522212.2', '121001.83+565716.7', '125453.81+291114.8', '160730.54+491932.4',
#                 '142214.08+531516.7', '163639.06+320400.0', '113535.74+533407.4', '141546.75-005604.2', '145206.22+331626.7', '222135.24+253943.1', '154059.00+401232.1',
#                 '135544.25+531805.2', '141758.85+324559.2', '141543.55+351620.1', '222831.07+274417.7', '223853.08+295530.5', '133948.78+013304.0', '161540.52+325720.1',
#                 '150717.25+255144.6', '144952.01+333031.6', '145806.56+355911.2', '164837.68+311652.7', '170809.44+211519.9', '211104.31-000747.3', '170254.81+244617.2',
#                 '161249.28+312523.0', '160524.52+303246.6', '154942.78+294506.1', '151639.06+280520.4', '122118.05+553355.8', '165335.83+354855.3', '165533.47+354942.7',
#                 '115625.26+270312.0', '120432.68+531311.1', '124151.80+534351.3', '122702.40+550531.9', '112838.87+501333.6', '142349.72+523903.6', '160833.97+421413.4',
#                 '153849.63+440637.7', '013620.77+301949.3', '134003.80+312424.5', '141956.38+510244.3', '023324.70-012819.6', '115837.97+001758.7', '122737.44+310439.5',
#                 '122256.17+555533.3']

def flux(mag, k, wavel): # k is the zero magnitude flux density. For W1 & W2, taken from a data table on the search website - https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html
        k = (k*(10**(-6))*(c*10**(10)))/(wavel**2) # converting from Jansky to 10-17 ergs/s/cm2/Å. Express c in Angstrom units
        return k*10**(-mag/2.5)

W1_k = 309.540 #Janskys. This means that mag 0 = 309.540 Janskys at the W1 wl.
W2_k = 171.787
W1_wl = 3.4e4 #Angstroms
W2_wl = 4.6e4

object_names_list = [] #Keeps track of objects that met MIR data requirements to take z score & absolute change

# z_score & absolute change lists
W1_SDSS_DESI = []
W1_SDSS_DESI_unc = []
W1_DESI_SDSS = []
W1_DESI_SDSS_unc = []
W1_abs_change = []
W1_abs_change_unc = []
W1_abs_change_norm = []
W1_abs_change_norm_unc = []
W1_dps = []

W2_SDSS_DESI = []
W2_SDSS_DESI_unc = []
W2_DESI_SDSS = []
W2_DESI_SDSS_unc = []
W2_abs_change = []
W2_abs_change_unc = []
W2_abs_change_norm = []
W2_abs_change_norm_unc = []
W2_dps = []

Min_SNR = 3 #Options are 10, 3, or 2. #A (SNR>10), B (3<SNR<10) or C (2<SNR<3)
if Min_SNR == 10: #Select Min_SNR on line above.
    MIR_SNR = 'A'
elif Min_SNR == 3:
    MIR_SNR = 'B'
elif Min_SNR == 2:
    MIR_SNR = 'C'
else:
    print('select a valid min SNR - 10, 3 or 2.')

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

g = 0
for object_name in object_names:
    print(g)
    print(object_name)
    g += 1
    parent_sample = pd.read_csv('guo23_parent_sample.csv')
    object_data = parent_sample[parent_sample.iloc[:, 4] == object_name]
    SDSS_RA = object_data.iloc[0, 1]
    SDSS_DEC = object_data.iloc[0, 2]
    SDSS_mjd = object_data.iloc[0, 6]
    DESI_mjd = object_data.iloc[0, 12]

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

    filtered_NEO_rows = NEO_data[(NEO_data.iloc[:, 36] > 5) & (NEO_data.iloc[:, 38] > 5)] #checking for rows where qual_frame is > 5 & separation of the WISE instrument to the South Atlantic Anomaly is > 5 degrees
    #"Single-exposure source database entries having qual_frame=0 should be used with extreme caution" - from the column descriptions.
    # The qi_fact column seems to be equal to qual_frame/10.

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
    W1_unc = filtered_WISE_rows.iloc[:, 12].tolist() + filtered_NEO_rows_W1.iloc[:, 19].tolist()
    W1_mag = list(zip(W1_mag, mjd_date_W1, W1_unc))
    W1_mag = [tup for tup in W1_mag if not np.isnan(tup[0])] #removing instances where the mag value is NaN

    mjd_date_W2 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W2.iloc[:, 42].tolist()
    W2_mag = filtered_WISE_rows.iloc[:, 14].tolist() + filtered_NEO_rows_W2.iloc[:, 22].tolist()
    W2_unc = filtered_WISE_rows.iloc[:, 15].tolist() + filtered_NEO_rows_W2.iloc[:, 23].tolist()
    W2_mag = list(zip(W2_mag, mjd_date_W2, W2_unc))
    W2_mag = [tup for tup in W2_mag if not np.isnan(tup[0])]

    if len(W1_mag) < 100: #want 100 data points as a minimum
        print('less than 100 W1')
        continue
    elif len(W2_mag) < 100:
        print('less than 100 W2')
        continue

    #Below code sorts MIR data.
    #One assumptions required for code to work:
    #1. The data is in order of oldest mjd to most recent.

    # W1 data first
    W1_list = []
    W1_unc_list = []
    W1_mjds = []
    W1_averages= []
    W1_av_uncs = []
    W1_av_mjd_date = []
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
            W1_list = []
            W1_mjds = []
            W1_unc_list = []
            W1_list.append(W1_mag[i][0])
            W1_mjds.append(W1_mag[i][1])
            W1_unc_list.append(W1_mag[i][2])
            continue

    # W2 data second
    W2_list = []
    W2_unc_list = []
    W2_mjds = []
    W2_averages= []
    W2_av_uncs = []
    W2_av_mjd_date = []
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
            W2_list = []
            W2_mjds = []
            W2_unc_list = []
            W2_list.append(W2_mag[i][0])
            W2_mjds.append(W2_mag[i][1])
            W2_unc_list.append(W2_mag[i][2])
            continue

    #Changing mjd date to days since start:
    min_mjd = min([W1_av_mjd_date[0], W2_av_mjd_date[0]])
    SDSS_mjd = SDSS_mjd - min_mjd
    DESI_mjd = DESI_mjd - min_mjd
    W1_av_mjd_date = [date - min_mjd for date in W1_av_mjd_date]
    W2_av_mjd_date = [date - min_mjd for date in W2_av_mjd_date]

    W1_averages_flux = [flux(mag, W1_k, W1_wl) for mag in W1_averages]
    W2_averages_flux = [flux(mag, W2_k, W2_wl) for mag in W2_averages]
    W1_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W1_av_uncs, W1_averages_flux)] #See document in week 5 folder for conversion.
    W2_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W2_av_uncs, W2_averages_flux)]

    before_SDSS_index_W1, after_SDSS_index_W1, q = find_closest_indices(W1_av_mjd_date, SDSS_mjd)
    before_SDSS_index_W2, after_SDSS_index_W2, w = find_closest_indices(W2_av_mjd_date, SDSS_mjd)
    before_DESI_index_W1, after_DESI_index_W1, e = find_closest_indices(W1_av_mjd_date, DESI_mjd)
    before_DESI_index_W2, after_DESI_index_W2, r = find_closest_indices(W2_av_mjd_date, DESI_mjd)

    if q == 0 and w == 0 and e == 0 and r == 0: #confirming that SDSS & DESI observations lie within the MIR observations
        # eliminating objects where there are 2 or more missing epochs around the SDSS & DESI observations.
        # if W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1] > 400:
        #     print('400 day gap')
        #     continue
        # elif W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2] > 400:
        #     print('400 day gap')
        #     continue
        # elif W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1] > 400:
        #     print('400 day gap')
        #     continue
        # elif W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2] > 400:
        #     print('400 day gap')
        #     continue

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

        object_names_list.append(object_name)

        #If uncertainty = nan; then z score = nan
        #If uncertainty = 0; then z score = inf
        W1_SDSS_DESI.append(W1_z_score_SDSS_DESI)
        W1_SDSS_DESI_unc.append(W1_z_score_SDSS_DESI_unc)
        W1_DESI_SDSS.append(W1_z_score_DESI_SDSS)
        W1_DESI_SDSS_unc.append(W1_z_score_DESI_SDSS_unc)
        W1_abs_change.append(W1_abs)
        W1_abs_change_unc.append(W1_abs_unc)
        W1_abs_change_norm.append(W1_abs_norm)
        W1_abs_change_norm_unc.append(W1_abs_norm_unc)
        W1_dps.append(len(W1_mag))

        W2_SDSS_DESI.append(W2_z_score_SDSS_DESI)
        W2_SDSS_DESI_unc.append(W2_z_score_SDSS_DESI_unc)
        W2_DESI_SDSS.append(W2_z_score_DESI_SDSS)
        W2_DESI_SDSS_unc.append(W2_z_score_DESI_SDSS_unc)
        W2_abs_change.append(W2_abs)
        W2_abs_change_unc.append(W2_abs_unc)
        W2_abs_change_norm.append(W2_abs_norm)
        W2_abs_change_norm_unc.append(W2_abs_norm_unc)
        W2_dps.append(len(W2_mag))

        SDSS_plate_number = object_data.iloc[0, 5]
        SDSS_fiberid_number = object_data.iloc[0, 7]
        SDSS_z = object_data.iloc[0, 3]
        DESI_z = object_data.iloc[0, 10]
        DESI_name = object_data.iloc[0, 11]
        SDSS_mjd_for_dnl = object_data.iloc[0, 6] #exact same as SDSS_mjd above, but that mjd gets converted to days since first observation

        #Automatically querying the SDSS database
        downloaded_SDSS_spec = SDSS.get_spectra_async(plate=SDSS_plate_number, fiberID=SDSS_fiberid_number, mjd=SDSS_mjd_for_dnl)
        downloaded_SDSS_spec = downloaded_SDSS_spec[0]

        hdul = HDUList(downloaded_SDSS_spec.get_fits())
        subset = hdul[1]

        #SDSS Spectrum information
        sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
        sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms

        #DESI spectrum retrieval method
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

        sfd = sfdmap.SFDMap('SFD_dust_files') #it says SFD - but the values are the same as S&F - I have checked for multiple objects
        ebv = sfd.ebv(coord)

        ext_model = G23(Rv=3.1) #Rv=3.1 is typical for MW - Schultz, Wiemer, 1975
        inverse_SDSS_lamb = [1/(x*10**(-4)) for x in sdss_lamb] #need units of inverse microns for extinguishing
        inverse_DESI_lamb = [1/(x*10**(-4)) for x in desi_lamb]
        sdss_flux = sdss_flux/ext_model.extinguish(inverse_SDSS_lamb, Ebv=ebv) #divide to remove the effect of dust
        desi_flux = desi_flux/ext_model.extinguish(inverse_DESI_lamb, Ebv=ebv)

        sdss_lamb = (sdss_lamb/(1+SDSS_z))
        desi_lamb = (desi_lamb/(1+DESI_z))

        gaussian_kernel = Gaussian1DKernel(stddev=3)

        # Smooth the flux data using the Gaussian kernel
        Gaus_smoothed_SDSS = convolve(sdss_flux, gaussian_kernel)
        if len(desi_flux) > 0:
            Gaus_smoothed_DESI = convolve(desi_flux, gaussian_kernel)
        else:
            Gaus_smoothed_DESI = []
        
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
        
        # Making a big figure with flux & SDSS, DESI spectra added in
        fig = plt.figure(figsize=(12, 7)) # (width, height)
        gs = GridSpec(5, 2, figure=fig)  # 5 rows, 2 columns

        # Top plot spanning two columns and three rows (ax1)
        ax1 = fig.add_subplot(gs[0:3, :])  # Rows 0 to 2, both columns
        ax1.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color='blue', capsize=5, label=u'W2 (4.6 \u03bcm)')
        ax1.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color='orange', capsize=5, label=u'W1 (3.4 \u03bcm)')
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
        ax2.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
        ax2.set_title('Gaussian Smoothed Plot of SDSS Spectrum')
        ax2.legend(loc='upper right')

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
        ax3.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
        ax3.set_title('Gaussian Smoothed Plot of DESI Spectrum')
        ax3.legend(loc='upper right')

        fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=1.25, wspace=0.2)
        #top and bottom adjust the vertical space on the top and bottom of the figure.
        #left and right adjust the horizontal space on the left and right sides.
        #hspace and wspace adjust the spacing between rows and columns, respectively.

        # fig.savefig(f'./CLAGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
        fig.savefig(f'./CLAGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')

    else:
        print('SDSS or DESI lie outside MIR observations in W1 or W2')
        continue

#for loop now ended
quantifying_change_data = {
    "Object": object_names_list,

    "W1 Z Score SDSS vs DESI": W1_SDSS_DESI,
    "W1 Z Score SDSS vs DESI Unc": W1_SDSS_DESI_unc,
    "W1 Z Score DESI vs SDSS": W1_DESI_SDSS,
    "W1 Z Score DESI vs SDSS Unc": W1_DESI_SDSS_unc,
    "W1 Flux Change": W1_abs_change,
    "W1 Flux Change Unc": W1_abs_change_unc,
    "W1 Normalised Flux Change": W1_abs_change_norm,
    "W1 Normalised Flux Change Unc": W1_abs_change_norm_unc,

    "W2 Z Score SDSS vs DESI": W2_SDSS_DESI,
    "W2 Z Score SDSS vs DESI Unc": W2_SDSS_DESI_unc,
    "W2 Z Score DESI vs SDSS": W2_DESI_SDSS,
    "W2 Z Score DESI vs SDSS Unc": W2_DESI_SDSS_unc,
    "W2 Flux Change": W2_abs_change,
    "W2 Flux Change Unc": W2_abs_change_unc,
    "W2 Normalised Flux Change": W2_abs_change_norm,
    "W2 Normalised Flux Change Unc": W2_abs_change_norm_unc,

    "W1 Data Points": W1_dps,
    "W2 Data Points": W2_dps,
}

# Convert the data into a DataFrame
df = pd.DataFrame(quantifying_change_data)

#Creating a csv file of my data
# df.to_csv("CLAGN_Quantifying_Change.csv", index=False)
df.to_csv("CLAGN_Quantifying_Change.csv", index=False)