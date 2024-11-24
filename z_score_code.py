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
object_names = ['085817.56+322349.7', '130115.40+252726.3', '101834.35+331258.9', '150210.72+522212.2', '121001.83+565716.7', '125453.81+291114.8', '160730.54+491932.4',
                '142214.08+531516.7', '163639.06+320400.0', '113535.74+533407.4', '141546.75-005604.2', '145206.22+331626.7', '222135.24+253943.1', '154059.00+401232.1',
                '135544.25+531805.2', '141758.85+324559.2', '141543.55+351620.1', '222831.07+274417.7', '223853.08+295530.5', '133948.78+013304.0', '161540.52+325720.1',
                '150717.25+255144.6', '144952.01+333031.6', '145806.56+355911.2', '164837.68+311652.7', '170809.44+211519.9', '211104.31-000747.3', '170254.81+244617.2',
                '161249.28+312523.0', '160524.52+303246.6', '154942.78+294506.1', '151639.06+280520.4', '122118.05+553355.8', '165335.83+354855.3', '165533.47+354942.7',
                '115625.26+270312.0']

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

object_names_list = [] #Keeps track of objects that met MIR data requirements to take z score & absolute change

# z_score & absolute change lists
W1_SDSS_DESI = []
W1_DESI_SDSS = []
W1_abs_change = []
W1_abs_change_norm = []

W2_SDSS_DESI = []
W2_DESI_SDSS = []
W2_abs_change = []
W2_abs_change_norm = []

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

    if len(W1_mag) < 50: #want 50 data points as a minimum
        continue
    elif len(W2_mag) < 50:
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


    if q == 0 and w == 0 and e == 0 and r == 0:

        #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to SDSS & DESI.
        W1_SDSS_interp = np.interp(SDSS_mjd, W1_av_mjd_date, W1_averages_flux)
        W2_SDSS_interp = np.interp(SDSS_mjd, W2_av_mjd_date, W2_averages_flux)
        W1_DESI_interp = np.interp(DESI_mjd, W1_av_mjd_date, W1_averages_flux)
        W2_DESI_interp = np.interp(DESI_mjd, W2_av_mjd_date, W2_averages_flux)

        W1_SDSS_unc_interp = np.sqrt(((((W1_av_mjd_date[after_SDSS_index_W1] - SDSS_mjd))/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[before_SDSS_index_W1])**2 + ((((SDSS_mjd - W1_av_mjd_date[before_SDSS_index_W1]))/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[after_SDSS_index_W1])**2)
        W2_SDSS_unc_interp = np.sqrt(((((W2_av_mjd_date[after_SDSS_index_W2] - SDSS_mjd))/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[before_SDSS_index_W2])**2 + ((((SDSS_mjd - W2_av_mjd_date[before_SDSS_index_W2]))/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[after_SDSS_index_W2])**2)
        W1_DESI_unc_interp = np.sqrt(((((W1_av_mjd_date[after_DESI_index_W1] - DESI_mjd))/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[before_DESI_index_W1])**2 + ((((DESI_mjd - W1_av_mjd_date[before_DESI_index_W1]))/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[after_DESI_index_W1])**2)
        W2_DESI_unc_interp = np.sqrt(((((W2_av_mjd_date[after_DESI_index_W2] - DESI_mjd))/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[before_DESI_index_W2])**2 + ((((DESI_mjd - W2_av_mjd_date[before_DESI_index_W2]))/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[after_DESI_index_W2])**2)
        
        object_names_list.append(object_name)

        #If uncertainty = nan; then z score = nan
        #If uncertainty = 0; then z score = inf
        W1_SDSS_DESI.append((W1_SDSS_interp-W1_DESI_interp)/(W1_DESI_unc_interp))
        W1_DESI_SDSS.append((W1_DESI_interp-W1_SDSS_interp)/(W1_SDSS_unc_interp))
        W1_abs_change.append(abs(W1_SDSS_interp-W1_DESI_interp)) #normalise this with the mean/median flux value?
        W1_abs_change_norm.append(abs((W1_SDSS_interp-W1_DESI_interp))/(np.average(W1_averages_flux)))

        W2_SDSS_DESI.append((W2_SDSS_interp-W2_DESI_interp)/(W2_DESI_unc_interp))
        W2_DESI_SDSS.append((W2_DESI_interp-W2_SDSS_interp)/(W2_SDSS_unc_interp))
        W2_abs_change.append(abs(W2_SDSS_interp-W2_DESI_interp))
        W2_abs_change_norm.append(abs((W2_SDSS_interp-W2_DESI_interp)/(np.average(W2_averages_flux))))

    else:
        continue

#for loop now ended
z_score_data = {
    "Object": object_names_list,

    "W1 Z Score SDSS vs DESI": W1_SDSS_DESI,
    "W1 Z Score DESI vs SDSS": W1_DESI_SDSS,
    "W1 Flux Change": W1_abs_change,
    "W1 Normalised Flux Change": W1_abs_change_norm,

    "W2 Z Score SDSS vs DESI": W2_SDSS_DESI,
    "W2 Z Score DESI vs SDSS": W2_DESI_SDSS,
    "W2 Flux Change": W2_abs_change,
    "W2 Normalised Flux Change": W2_abs_change_norm,
}

# Convert the data into a DataFrame
df = pd.DataFrame(z_score_data)

#Creating a csv file of my data
df.to_csv("CLAGN_Quantifying_Change.csv", index=False)