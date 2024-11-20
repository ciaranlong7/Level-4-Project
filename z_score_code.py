import numpy as np
import pandas as pd
from astropy import units as u #In Astropy, a Quantity object combines a numerical value (like a 1D array of flux) with a physical unit (like W/m^2, erg/s, etc.)
from astropy.visualization import quantity_support
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa
quantity_support()  # for getting units on the axes below

c = 299792458

Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

# #random list of object names taken from parent catalogue
# object_names = ['085817.56+322349.7', '130115.40+252726.3', '101834.35+331258.9', '150210.72+522212.2', '121001.83+565716.7', '125453.81+291114.8', '160730.54+491932.4',
#                 '142214.08+531516.7', '163639.06+320400.0', '113535.74+533407.4', '141546.75-005604.2', '145206.22+331626.7', '222135.24+253943.1', '154059.00+401232.1',
#                 '135544.25+531805.2', '141758.85+324559.2', '141543.55+351620.1', '222831.07+274417.7', '223853.08+295530.5', '133948.78+013304.0', '161540.52+325720.1']

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

object_names_list = [] #Keeps track of objects that had the right data to actual take z_scores

# z_score_lists
W1_b_SDSS_b_DESI = []
W1_a_SDSS_b_DESI = []
W1_b_SDSS_a_DESI = []
W1_a_SDSS_a_DESI = []
W1_b_DESI_b_SDSS = []
W1_a_DESI_b_SDSS = []
W1_b_DESI_a_SDSS = []
W1_a_DESI_a_SDSS = []

W2_b_SDSS_b_DESI = []
W2_a_SDSS_b_DESI = []
W2_b_SDSS_a_DESI = []
W2_a_SDSS_a_DESI = []
W2_b_DESI_b_SDSS = []
W2_a_DESI_b_SDSS = []
W2_b_DESI_a_SDSS = []
W2_a_DESI_a_SDSS = []

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

    mjd_date_W2 = filtered_WISE_rows.iloc[:, 10].tolist() + filtered_NEO_rows_W2.iloc[:, 42].tolist()
    W2_mag = filtered_WISE_rows.iloc[:, 14].tolist() + filtered_NEO_rows_W2.iloc[:, 22].tolist()
    W2_unc = filtered_WISE_rows.iloc[:, 15].tolist() + filtered_NEO_rows_W2.iloc[:, 23].tolist()
    W2_mag = list(zip(W2_mag, mjd_date_W2, W2_unc))

    if len(W1_mag) < 50: #want 50 data points as a minimum
        continue
    elif len(W2_mag) < 50:
        continue

    #Below code sorts MIR data.
    #Two assumptions required for code to work:
    #1. There is never a situation where the data has only one data point for an epoch.
    #2. The data is in order of oldest mjd to most recent.

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

    # Selecting the 2 points either side of SDSS & DESI
    if SDSS_mjd <= W1_av_mjd_date[0]:
        # print("SDSS observation was before WISE observation.")
        continue
    elif SDSS_mjd >= W1_av_mjd_date[-1]:
        # print("SDSS observation was after WISE observation.") #Not possible
        continue
    elif SDSS_mjd <= W2_av_mjd_date[0]:
        continue
    elif SDSS_mjd >= W2_av_mjd_date[-1]:
        continue
    else:
        before_SDSS_index_W1 = max(i for i in range(len(W1_av_mjd_date)) if W1_av_mjd_date[i] <= SDSS_mjd) #different for W1 & W2 in case there are a different number of W1 & W2 epochs
        after_SDSS_index_W1 = min(i for i in range(len(W1_av_mjd_date)) if W1_av_mjd_date[i] > SDSS_mjd)
        before_SDSS_index_W2 = max(i for i in range(len(W2_av_mjd_date)) if W2_av_mjd_date[i] <= SDSS_mjd)
        after_SDSS_index_W2 = min(i for i in range(len(W2_av_mjd_date)) if W2_av_mjd_date[i] > SDSS_mjd)

    if DESI_mjd <= W1_av_mjd_date[0]:
        # print("DESI observation was before WISE observation.") #Not possible
        continue
    elif DESI_mjd >= W1_av_mjd_date[-1]:
        # print("DESI observation was after WISE observation.")
        continue
    elif DESI_mjd <= W2_av_mjd_date[0]:
        continue
    elif DESI_mjd >= W2_av_mjd_date[-1]:
        continue
    else:
        before_DESI_index_W1 = max(i for i in range(len(W1_av_mjd_date)) if W1_av_mjd_date[i] <= DESI_mjd)
        after_DESI_index_W1 = min(i for i in range(len(W1_av_mjd_date)) if W1_av_mjd_date[i] > DESI_mjd)
        before_DESI_index_W2 = max(i for i in range(len(W2_av_mjd_date)) if W2_av_mjd_date[i] <= DESI_mjd)
        after_DESI_index_W2 = min(i for i in range(len(W2_av_mjd_date)) if W2_av_mjd_date[i] > DESI_mjd)

    W1_averages_flux = [flux(mag, W1_k, W1_wl) for mag in W1_averages]
    W2_averages_flux = [flux(mag, W2_k, W2_wl) for mag in W2_averages]
    W1_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W1_av_uncs, W1_averages_flux)] #See document in week 5 folder for conversion.
    W2_av_uncs_flux = [((unc*np.log(10))/(2.5))*flux for unc, flux in zip(W2_av_uncs, W2_averages_flux)]

    object_names_list.append(object_name)

    #If uncertainty = nan; then z score = nan
    #If uncertainty = 0; then z score = inf
    W1_b_SDSS_b_DESI.append((W1_averages_flux[before_SDSS_index_W1]-W1_averages_flux[before_DESI_index_W1])/(W1_av_uncs_flux[before_DESI_index_W1]))
    W1_a_SDSS_b_DESI.append((W1_averages_flux[after_SDSS_index_W1]-W1_averages_flux[before_DESI_index_W1])/(W1_av_uncs_flux[before_DESI_index_W1]))
    W1_b_SDSS_a_DESI.append((W1_averages_flux[before_SDSS_index_W1]-W1_averages_flux[after_DESI_index_W1])/(W1_av_uncs_flux[after_DESI_index_W1]))
    W1_a_SDSS_a_DESI.append((W1_averages_flux[after_SDSS_index_W1]-W1_averages_flux[after_DESI_index_W1])/(W1_av_uncs_flux[after_DESI_index_W1]))
    W1_b_DESI_b_SDSS.append((W1_averages_flux[before_DESI_index_W1]-W1_averages_flux[before_SDSS_index_W1])/(W1_av_uncs_flux[before_SDSS_index_W1]))
    W1_a_DESI_b_SDSS.append((W1_averages_flux[after_DESI_index_W1]-W1_averages_flux[before_SDSS_index_W1])/(W1_av_uncs_flux[before_SDSS_index_W1]))
    W1_b_DESI_a_SDSS.append((W1_averages_flux[before_DESI_index_W1]-W1_averages_flux[after_SDSS_index_W1])/(W1_av_uncs_flux[after_SDSS_index_W1]))
    W1_a_DESI_a_SDSS.append((W1_averages_flux[after_DESI_index_W1]-W1_averages_flux[after_SDSS_index_W1])/(W1_av_uncs_flux[after_SDSS_index_W1]))

    W2_b_SDSS_b_DESI.append((W2_averages_flux[before_SDSS_index_W2]-W2_averages_flux[before_DESI_index_W2])/(W2_av_uncs_flux[before_DESI_index_W2]))
    W2_a_SDSS_b_DESI.append((W2_averages_flux[after_SDSS_index_W2]-W2_averages_flux[before_DESI_index_W2])/(W2_av_uncs_flux[before_DESI_index_W2]))
    W2_b_SDSS_a_DESI.append((W2_averages_flux[before_SDSS_index_W2]-W2_averages_flux[after_DESI_index_W2])/(W2_av_uncs_flux[after_DESI_index_W2]))
    W2_a_SDSS_a_DESI.append((W2_averages_flux[after_SDSS_index_W2]-W2_averages_flux[after_DESI_index_W2])/(W2_av_uncs_flux[after_DESI_index_W2]))
    W2_b_DESI_b_SDSS.append((W2_averages_flux[before_DESI_index_W2]-W2_averages_flux[before_SDSS_index_W2])/(W2_av_uncs_flux[before_SDSS_index_W2]))
    W2_a_DESI_b_SDSS.append((W2_averages_flux[after_DESI_index_W2]-W2_averages_flux[before_SDSS_index_W2])/(W2_av_uncs_flux[before_SDSS_index_W2]))
    W2_b_DESI_a_SDSS.append((W2_averages_flux[before_DESI_index_W2]-W2_averages_flux[after_SDSS_index_W2])/(W2_av_uncs_flux[after_SDSS_index_W2]))
    W2_a_DESI_a_SDSS.append((W2_averages_flux[after_DESI_index_W2]-W2_averages_flux[after_SDSS_index_W2])/(W2_av_uncs_flux[after_SDSS_index_W2]))

#for loop now ended
z_score_data = {
    "Object": object_names_list,
    "W1 Before SDSS vs Before DESI": W1_b_SDSS_b_DESI,
    "W1 After SDSS vs Before DESI": W1_a_SDSS_b_DESI,
    "W1 Before SDSS vs After DESI": W1_b_SDSS_a_DESI,
    "W1 After SDSS vs After DESI": W1_a_SDSS_a_DESI,
    "W1 Before DESI vs Before SDSS": W1_b_DESI_b_SDSS,
    "W1 After DESI vs Before SDSS": W1_a_DESI_b_SDSS,
    "W1 Before DESI vs After SDSS": W1_b_DESI_a_SDSS,
    "W1 After DESI vs After SDSS": W1_a_DESI_a_SDSS,

    "W2 Before SDSS vs Before DESI": W2_b_SDSS_b_DESI,
    "W2 After SDSS vs Before DESI": W2_a_SDSS_b_DESI,
    "W2 Before SDSS vs After DESI": W2_b_SDSS_a_DESI,
    "W2 After SDSS vs After DESI": W2_a_SDSS_a_DESI,
    "W2 Before DESI vs Before SDSS": W2_b_DESI_b_SDSS,
    "W2 After DESI vs Before SDSS": W2_a_DESI_b_SDSS,
    "W2 Before DESI vs After SDSS": W2_b_DESI_a_SDSS,
    "W2 After DESI vs After SDSS": W2_a_DESI_a_SDSS,
}

# Convert the data into a DataFrame
df = pd.DataFrame(z_score_data)

#Creating a csv file of my data
df.to_csv("CLAGN_z_scores.csv", index=False)