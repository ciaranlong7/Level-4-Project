import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.interpolate import interp1d
from astropy.io import fits
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

# #When changing object names list from CLAGN to AGN - I must change the files I am saving to at the bottom as well.
# parent_sample = pd.read_csv('guo23_parent_sample.csv')
# columns_to_check = parent_sample.columns[[3, 5, 11]] #removing duplicates where SDSS name, SDSS mjd & DESI mjd all the same
# parent_sample = parent_sample.drop_duplicates(subset=columns_to_check)
# object_names = parent_sample.iloc[:, 4].sample(n=400, random_state=42) #randomly selecting 250 object names from parent sample

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
W1_SDSS_bef_dps = []
W1_SDSS_aft_dps = []
W1_DESI_bef_dps = []
W1_DESI_aft_dps = []
W1_SDSS_gap = []
W1_DESI_gap = []

W2_SDSS_DESI = []
W2_SDSS_DESI_unc = []
W2_DESI_SDSS = []
W2_DESI_SDSS_unc = []
W2_abs_change = []
W2_abs_change_unc = []
W2_abs_change_norm = []
W2_abs_change_norm_unc = []
W2_SDSS_bef_dps = []
W2_SDSS_aft_dps = []
W2_DESI_bef_dps = []
W2_DESI_aft_dps = []
W2_SDSS_gap = []
W2_DESI_gap = []

mean_zscore = []
mean_zscore_unc = []
mean_norm_flux_change = []
mean_norm_flux_change_unc = []
mean_UV_flux_change = []
mean_UV_flux_change_unc = []

Min_SNR = 3 #Options are 10, 3, or 2. #A (SNR>10), B (3<SNR<10) or C (2<SNR<3)
if Min_SNR == 10: #Select Min_SNR on line above.
    MIR_SNR = 'A'
elif Min_SNR == 3:
    MIR_SNR = 'B'
elif Min_SNR == 2:
    MIR_SNR = 'C'
else:
    print('select a valid min SNR - 10, 3 or 2.')

max_day_gap = 600 #max day gap to linearly interpolate over
min_dps = 1 #minimum dps per epoch

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
            if x_vals[after_index] - x_vals[before_index] > max_day_gap:
                t += 1
            return before_index, after_index, t

#DESI spectrum retrieval method
def get_primary_spectrum(targetid): #some objects have multiple spectra for it in DESI- the best one is the 'primary' spectrum
    """
    Retrieves the primary spectrum's wavelength and flux for a given target ID.

    Parameters:
    - targetid (int): The target ID of the object.

    Returns:
    - desi_lamb (array): Wavelength data for the primary spectrum.
    - desi_flux (array): Flux data for the primary spectrum.
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
        print(f'DESI Spectrum cannot be found for object_name = {object_name}, DESI target_id = {DESI_name}')

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
        print(f'DESI Spectrum cannot be found for object_name = {object_name}, DESI target_id = {DESI_name}')

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

sfd = sfdmap.SFDMap('SFD_dust_files') #called SFD map, but see - https://github.com/kbarbary/sfdmap/blob/master/README.md
# It explains how "By default, a scaling of 0.86 is applied to the map values to reflect the recalibration by Schlafly & Finkbeiner (2011)"
ext_model = G23(Rv=3.1) #Rv=3.1 is typical for MW - Schultz, Wiemer, 1975
gaussian_kernel = Gaussian1DKernel(stddev=3)

parent_sample = pd.read_csv('guo23_parent_sample.csv')
parent_sample = parent_sample.iloc[:, 1:] #drop the first column (the index)
columns_to_check = parent_sample.columns[[3, 5, 11]] #checking SDSS name, SDSS mjd & DESI mjd
parent_sample = parent_sample.drop_duplicates(subset=columns_to_check)
g = 0
for object_name in object_names:
    print(g)
    print(object_name)
    g += 1
    object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
    SDSS_RA = object_data.iloc[0, 0]
    SDSS_DEC = object_data.iloc[0, 1]
    SDSS_mjd = object_data.iloc[0, 5]
    DESI_mjd = object_data.iloc[0, 11]

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

    if len(filtered_NEO_rows_W1.iloc[:, 42].tolist()) < 2: #checking if there is enough NEO data
        print('No W1 data')
        continue
    elif len(filtered_NEO_rows_W2.iloc[:, 42].tolist()) < 2:
        print('No W2 data')
        continue

    if len(filtered_WISE_rows.iloc[:, 10].tolist()) > 0: #checking if there is any ALLWISE data
        final_ALLWISE_mjd = filtered_WISE_rows.iloc[:, 10].tolist()[-1]
        first_NEO_mjd_W1 = filtered_NEO_rows_W1.iloc[:, 42].tolist()[0]
        first_NEO_mjd_W2 = filtered_NEO_rows_W2.iloc[:, 42].tolist()[0]

        if final_ALLWISE_mjd < SDSS_mjd < first_NEO_mjd_W1:
            print(f'SDSS observation was {SDSS_mjd-final_ALLWISE_mjd} from ALLWISE, {first_NEO_mjd_W1-SDSS_mjd} from NEOWISE W1.')
            continue
        elif final_ALLWISE_mjd < SDSS_mjd < first_NEO_mjd_W2:
            print(f'SDSS observation was {SDSS_mjd-final_ALLWISE_mjd} from ALLWISE, {first_NEO_mjd_W2-SDSS_mjd} from NEOWISE W2.')
            continue
    else:
        pass #already have code to filter out objects with SDSS before 1st WISE. this code is just to fileter out if sdss in all-neo gap.

    #Below code sorts MIR data.
    #Two assumptions required for code to work:
    #1. The data is sorted in order of oldest mjd to most recent.
    #2. There are 2 or more data points.

    # W1 data first
    W1_list = []
    W1_unc_list = []
    W1_mjds = []
    W1_averages= []
    W1_av_uncs = []
    W1_av_mjd_date = []
    W1_epoch_dps = []  
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
            W1_averages.append(np.median(W1_list))
            W1_av_mjd_date.append(np.median(W1_mjds))
            W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
            W1_epoch_dps.append(len(W1_list))
            continue
        elif W1_mag[i][1] - W1_mag[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
            W1_list.append(W1_mag[i][0])
            W1_mjds.append(W1_mag[i][1])
            W1_unc_list.append(W1_mag[i][2])
            continue
        else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
            W1_averages.append(np.median(W1_list))
            W1_av_mjd_date.append(np.median(W1_mjds))
            W1_av_uncs.append((1/len(W1_unc_list))*np.sqrt(np.sum(np.square(W1_unc_list))))
            W1_epoch_dps.append(len(W1_list))
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
    W2_epoch_dps = []  
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
            W2_averages.append(np.median(W2_list))
            W2_av_mjd_date.append(np.median(W2_mjds))
            W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
            W2_epoch_dps.append(len(W2_list))
            continue
        elif W2_mag[i][1] - W2_mag[i-1][1] < 100: #checking in the same epoch (<100 days between measurements)
            W2_list.append(W2_mag[i][0])
            W2_mjds.append(W2_mag[i][1])
            W2_unc_list.append(W2_mag[i][2])
            continue
        else: #if the gap is bigger than 100 days, then take the averages and reset the lists.
            W2_averages.append(np.median(W2_list))
            W2_av_mjd_date.append(np.median(W2_mjds))
            W2_av_uncs.append((1/len(W2_unc_list))*np.sqrt(np.sum(np.square(W2_unc_list))))
            W2_epoch_dps.append(len(W2_list))
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

    #Filtering for min dps in all adjactent epochs
    if W1_epoch_dps[before_SDSS_index_W1] < min_dps:
        if W2_epoch_dps[before_SDSS_index_W2] < min_dps or W2_epoch_dps[after_SDSS_index_W2] < min_dps or W2_epoch_dps[before_DESI_index_W2] < min_dps or W2_epoch_dps[after_DESI_index_W2] < min_dps: 
            print('Not enough data in W1 & W2')
            continue
    elif W1_epoch_dps[after_SDSS_index_W1] < min_dps:
        if W2_epoch_dps[before_SDSS_index_W2] < min_dps or W2_epoch_dps[after_SDSS_index_W2] < min_dps or W2_epoch_dps[before_DESI_index_W2] < min_dps or W2_epoch_dps[after_DESI_index_W2] < min_dps: 
            print('Not enough data in W1 & W2')
            continue
    elif W1_epoch_dps[before_DESI_index_W1] < min_dps:
        if W2_epoch_dps[before_SDSS_index_W2] < min_dps or W2_epoch_dps[after_SDSS_index_W2] < min_dps or W2_epoch_dps[before_DESI_index_W2] < min_dps or W2_epoch_dps[after_DESI_index_W2] < min_dps: 
            print('Not enough data in W1 & W2')
            continue
    elif W1_epoch_dps[after_DESI_index_W1] < min_dps:
        if W2_epoch_dps[before_SDSS_index_W2] < min_dps or W2_epoch_dps[after_SDSS_index_W2] < min_dps or W2_epoch_dps[before_DESI_index_W2] < min_dps or W2_epoch_dps[after_DESI_index_W2] < min_dps: 
            print('Not enough data in W1 & W2')
            continue

    SDSS_plate_number = object_data.iloc[0, 4]
    SDSS_plate = f'{SDSS_plate_number:04}'
    SDSS_mjd_for_dnl = object_data.iloc[0, 5] #exact same as SDSS_mjd above, but that mjd gets converted to days since first observation
    SDSS_fiberid_number = object_data.iloc[0, 6]
    SDSS_fiberid = f"{SDSS_fiberid_number:04}"
    SDSS_z = object_data.iloc[0, 2]
    DESI_z = object_data.iloc[0, 9]
    DESI_name = object_data.iloc[0, 10]

    #Automatically querying the SDSS database
    downloaded_SDSS_spec = SDSS.get_spectra_async(plate=SDSS_plate_number, fiberID=SDSS_fiberid_number, mjd=SDSS_mjd)
    if downloaded_SDSS_spec == None:
        downloaded_SDSS_spec = SDSS.get_spectra_async(coordinates=coord, radius=2. * u.arcsec)
        if downloaded_SDSS_spec == None:
            print(f'SDSS Spectrum cannot be found for object_name = {object_name}')
            try:
                SDSS_file = f'spec-{SDSS_plate}-{SDSS_mjd:.0f}-{SDSS_fiberid}.fits'
                SDSS_file_path = f'clagn_spectra/{SDSS_file}'
                with fits.open(SDSS_file_path) as hdul:
                    subset = hdul[1]

                    sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
                    sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
                    sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
                    print('SDSS file is in downloads - will proceed as normal')
            except FileNotFoundError as e:
                print('No SDSS file already downloaded.')
                sdss_flux = []
                sdss_lamb = []
        else:
            downloaded_SDSS_spec = downloaded_SDSS_spec[0]
            hdul = HDUList(downloaded_SDSS_spec.get_fits())
            subset = hdul[1]

            sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
            sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
            sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])
    else:
        downloaded_SDSS_spec = downloaded_SDSS_spec[0]
        hdul = HDUList(downloaded_SDSS_spec.get_fits())
        subset = hdul[1]

        sdss_flux = subset.data['flux'] # 10-17 ergs/s/cm2/Å
        sdss_lamb = 10**subset.data['loglam'] #Wavelength in Angstroms
        sdss_flux_unc = np.array([np.sqrt(1/val) if val!=0 else np.nan for val in subset.data['ivar']])

    target_id = int(DESI_name)
    desi_lamb, desi_flux = get_primary_spectrum(target_id)

    ebv = sfd.ebv(coord)
    inverse_SDSS_lamb = [1/(x*10**(-4)) for x in sdss_lamb] #need units of inverse microns for extinguishing
    inverse_DESI_lamb = [1/(x*10**(-4)) for x in desi_lamb]
    sdss_flux = sdss_flux/ext_model.extinguish(inverse_SDSS_lamb, Ebv=ebv) #divide to remove the effect of dust
    desi_flux = desi_flux/ext_model.extinguish(inverse_DESI_lamb, Ebv=ebv)

    sdss_lamb = (sdss_lamb/(1+SDSS_z))
    desi_lamb = (desi_lamb/(1+DESI_z))

    if len(sdss_flux) > 0:
        Gaus_smoothed_SDSS = convolve(sdss_flux, gaussian_kernel)
    else:
        Gaus_smoothed_SDSS = []
    if len(desi_flux) > 0:
        Gaus_smoothed_DESI = convolve(desi_flux, gaussian_kernel)
    else:
        Gaus_smoothed_DESI = []

    if len(sdss_lamb) > 0:
        SDSS_min = min(sdss_lamb)
        SDSS_max = max(sdss_lamb)
    else:
        SDSS_min = 0
        SDSS_max = 1
    if len(desi_lamb) > 0:
        DESI_min = min(desi_lamb)
        DESI_max = max(desi_lamb)
    else:
        DESI_min = 0
        DESI_max = 1

    if q == 0 and e == 0: #Good W1 if true
        if w == 0 and r == 0: #Good W2 if true
            #Good W1 & W2
            object_names_list.append(object_name)

            #Linearly interpolating to get interpolated flux on a value in between the data points adjacent to SDSS & DESI.
            W1_SDSS_interp = np.interp(SDSS_mjd, W1_av_mjd_date, W1_averages_flux)
            W1_DESI_interp = np.interp(DESI_mjd, W1_av_mjd_date, W1_averages_flux)

            #uncertainties in interpolated flux
            W1_SDSS_unc_interp = np.sqrt((((W1_av_mjd_date[after_SDSS_index_W1] - SDSS_mjd)/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[before_SDSS_index_W1])**2 + (((SDSS_mjd - W1_av_mjd_date[before_SDSS_index_W1])/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[after_SDSS_index_W1])**2)
            W1_DESI_unc_interp = np.sqrt((((W1_av_mjd_date[after_DESI_index_W1] - DESI_mjd)/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[before_DESI_index_W1])**2 + (((DESI_mjd - W1_av_mjd_date[before_DESI_index_W1])/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[after_DESI_index_W1])**2)

            #uncertainty in absolute flux change
            W1_abs = abs(W1_SDSS_interp-W1_DESI_interp)
            W1_abs_unc = np.sqrt(W1_SDSS_unc_interp**2 + W1_DESI_unc_interp**2)

            #uncertainty in normalised flux change
            W1_av_unc = (1/len(W1_av_uncs_flux))*np.sqrt(sum(unc**2 for unc in W1_av_uncs_flux)) #uncertainty of the mean flux value
            W1_abs_norm = ((W1_abs)/(np.median(W1_averages_flux)))
            W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_av_unc)/(np.median(W1_averages_flux)))**2)

            #uncertainty in z score
            W1_z_score_SDSS_DESI = (W1_SDSS_interp-W1_DESI_interp)/(W1_DESI_unc_interp)
            W1_z_score_SDSS_DESI_unc = W1_z_score_SDSS_DESI*((W1_abs_unc)/(W1_abs))
            W1_z_score_DESI_SDSS = (W1_DESI_interp-W1_SDSS_interp)/(W1_SDSS_unc_interp)
            W1_z_score_DESI_SDSS_unc = W1_z_score_DESI_SDSS*((W1_abs_unc)/(W1_abs))

            W1_SDSS_DESI.append(W1_z_score_SDSS_DESI)
            W1_SDSS_DESI_unc.append(W1_z_score_SDSS_DESI_unc)
            W1_DESI_SDSS.append(W1_z_score_DESI_SDSS)
            W1_DESI_SDSS_unc.append(W1_z_score_DESI_SDSS_unc)
            W1_abs_change.append(W1_abs)
            W1_abs_change_unc.append(W1_abs_unc)
            W1_abs_change_norm.append(W1_abs_norm)
            W1_abs_change_norm_unc.append(W1_abs_norm_unc)

            W1_SDSS_bef_dps.append(W1_epoch_dps[before_SDSS_index_W1])
            W1_SDSS_aft_dps.append(W1_epoch_dps[after_SDSS_index_W1])
            W1_DESI_bef_dps.append(W1_epoch_dps[before_DESI_index_W1])
            W1_DESI_aft_dps.append(W1_epoch_dps[after_DESI_index_W1])

            W1_SDSS_gap.append(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1])
            W1_DESI_gap.append(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1])

            W2_SDSS_interp = np.interp(SDSS_mjd, W2_av_mjd_date, W2_averages_flux)
            W2_DESI_interp = np.interp(DESI_mjd, W2_av_mjd_date, W2_averages_flux)

            W2_SDSS_unc_interp = np.sqrt((((W2_av_mjd_date[after_SDSS_index_W2] - SDSS_mjd)/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[before_SDSS_index_W2])**2 + (((SDSS_mjd - W2_av_mjd_date[before_SDSS_index_W2])/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[after_SDSS_index_W2])**2)
            W2_DESI_unc_interp = np.sqrt((((W2_av_mjd_date[after_DESI_index_W2] - DESI_mjd)/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[before_DESI_index_W2])**2 + (((DESI_mjd - W2_av_mjd_date[before_DESI_index_W2])/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[after_DESI_index_W2])**2)

            W2_abs = abs(W2_SDSS_interp-W2_DESI_interp)
            W2_abs_unc = np.sqrt(W2_SDSS_unc_interp**2 + W2_DESI_unc_interp**2)

            W2_av_unc = (1/len(W2_av_uncs_flux))*np.sqrt(sum(unc**2 for unc in W2_av_uncs_flux)) #uncertainty of the mean flux value
            W2_abs_norm = ((W2_abs)/(np.median(W2_averages_flux)))
            W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_av_unc)/(np.median(W2_averages_flux)))**2)

            W2_z_score_SDSS_DESI = (W2_SDSS_interp-W2_DESI_interp)/(W2_DESI_unc_interp)
            W2_z_score_SDSS_DESI_unc = W2_z_score_SDSS_DESI*((W2_abs_unc)/(W2_abs))
            W2_z_score_DESI_SDSS = (W2_DESI_interp-W2_SDSS_interp)/(W2_SDSS_unc_interp)
            W2_z_score_DESI_SDSS_unc = W2_z_score_DESI_SDSS*((W2_abs_unc)/(W2_abs))

            W2_SDSS_DESI.append(W2_z_score_SDSS_DESI)
            W2_SDSS_DESI_unc.append(W2_z_score_SDSS_DESI_unc)
            W2_DESI_SDSS.append(W2_z_score_DESI_SDSS)
            W2_DESI_SDSS_unc.append(W2_z_score_DESI_SDSS_unc)
            W2_abs_change.append(W2_abs)
            W2_abs_change_unc.append(W2_abs_unc)
            W2_abs_change_norm.append(W2_abs_norm)
            W2_abs_change_norm_unc.append(W2_abs_norm_unc)

            W2_SDSS_bef_dps.append(W2_epoch_dps[before_SDSS_index_W2])
            W2_SDSS_aft_dps.append(W2_epoch_dps[after_SDSS_index_W2])
            W2_DESI_bef_dps.append(W2_epoch_dps[before_DESI_index_W2])
            W2_DESI_aft_dps.append(W2_epoch_dps[after_DESI_index_W2])

            W2_SDSS_gap.append(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2])
            W2_DESI_gap.append(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2])

            zscores = np.sort([W1_z_score_SDSS_DESI, W1_z_score_DESI_SDSS, W2_z_score_SDSS_DESI, W2_z_score_DESI_SDSS]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_SDSS_DESI_unc, W1_z_score_DESI_SDSS_unc, W2_z_score_SDSS_DESI_unc, W2_z_score_DESI_SDSS_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(abs(zscores)))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(abs(zscores)))
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

            if SDSS_min < 3000 and SDSS_max > 3920 and DESI_min < 3000 and DESI_max > 3920:
                closest_index_lower_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
                closest_index_upper_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
                sdss_blue_lamb = sdss_lamb[closest_index_lower_sdss:closest_index_upper_sdss]
                sdss_blue_flux = sdss_flux[closest_index_lower_sdss:closest_index_upper_sdss]
                sdss_blue_flux_smooth = Gaus_smoothed_SDSS[closest_index_lower_sdss:closest_index_upper_sdss]

                desi_lamb = desi_lamb.tolist()
                closest_index_lower_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
                closest_index_upper_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
                desi_blue_lamb = desi_lamb[closest_index_lower_desi:closest_index_upper_desi]
                desi_blue_flux = desi_flux[closest_index_lower_desi:closest_index_upper_desi]
                desi_blue_flux_smooth = Gaus_smoothed_DESI[closest_index_lower_desi:closest_index_upper_desi]

                #interpolating SDSS flux so lambda values match up with DESI . Done this way round because DESI lambda values are closer together.
                sdss_interp_fn = interp1d(sdss_blue_lamb, sdss_blue_flux_smooth, kind='linear', fill_value='extrapolate')
                sdss_blue_flux_interp = sdss_interp_fn(desi_blue_lamb) #interpolating the sdss flux to be in line with the desi lambda values

                UV_flux_change = [desi - sdss for sdss, desi in zip(sdss_blue_flux_interp, desi_blue_flux_smooth)]

                mean_UV_flux_change.append(np.mean(UV_flux_change))
                mean_UV_flux_change_unc.append(np.std(UV_flux_change))
            else:
                mean_UV_flux_change.append(np.nan)
                mean_UV_flux_change_unc.append(np.nan)
            
            # #Uncomment out if I want to save plots for good W1 & W2 data
            # #BELs
            # H_alpha = 6562.819
            # H_beta = 4861.333
            # Mg2 = 2795.528
            # C3_ = 1908.734
            # C4 = 1548.187
            # Ly_alpha = 1215.670
            # Ly_beta = 1025.722
            # #NEL
            # _O3_ = 5006.843 #underscores indicate square brackets
            # #Note there are other [O III] lines, such as: 4958.911 A, 4363.210 A

            # # Making a big figure with flux & SDSS, DESI spectra added in
            # fig = plt.figure(figsize=(12, 7)) # (width, height)
            # gs = GridSpec(5, 2, figure=fig)  # 5 rows, 2 columns

            # # Top plot spanning two columns and three rows (ax1)
            # ax1 = fig.add_subplot(gs[0:3, :])  # Rows 0 to 2, both columns
            # ax1.errorbar(W2_av_mjd_date, W2_averages_flux, yerr=W2_av_uncs_flux, fmt='o', color='blue', capsize=5, label=u'W2 (4.6 \u03bcm)')
            # ax1.errorbar(W1_av_mjd_date, W1_averages_flux, yerr=W1_av_uncs_flux, fmt='o', color='orange', capsize=5, label=u'W1 (3.4 \u03bcm)')
            # ax1.axvline(SDSS_mjd, linewidth=2, color='forestgreen', linestyle='--', label='SDSS Observation')
            # ax1.axvline(DESI_mjd, linewidth=2, color='midnightblue', linestyle='--', label='DESI Observation')
            # ax1.set_xlabel('Days since first observation')
            # ax1.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
            # ax1.set_title(f'Flux vs Time ({object_name})')
            # ax1.legend(loc='best')

            # # Bottom left plot spanning 2 rows and 1 column (ax2)
            # ax2 = fig.add_subplot(gs[3:, 0])  # Rows 3 to 4, first column
            # ax2.plot(sdss_lamb, sdss_flux, alpha=0.2, color='forestgreen')
            # ax2.plot(sdss_lamb, Gaus_smoothed_SDSS, color='forestgreen')
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
            # # if SDSS_min <= _O3_ <= SDSS_max:
            # #     ax2.axvline(_O3_, linewidth=2, color='grey', label = '[O III]')
            # if SDSS_min <= Ly_alpha <= SDSS_max:
            #     ax2.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
            # if SDSS_min <= Ly_beta <= SDSS_max:
            #     ax2.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
            # ax2.set_xlabel('Wavelength / Å')
            # ax2.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
            # ax2.set_title('Gaussian Smoothed Plot of SDSS Spectrum')
            # ax2.legend(loc='upper right')

            # # Bottom right plot spanning 2 rows and 1 column (ax3)
            # ax3 = fig.add_subplot(gs[3:, 1])  # Rows 3 to 4, second column
            # ax3.plot(desi_lamb, desi_flux, alpha=0.2, color='midnightblue')
            # ax3.plot(desi_lamb, Gaus_smoothed_DESI, color='midnightblue')
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
            # # if DESI_min <= _O3_ <= DESI_max:
            # #     ax3.axvline(_O3_, linewidth=2, color='grey', label = '[O III]')
            # if DESI_min <= Ly_alpha <= DESI_max:
            #     ax3.axvline(Ly_alpha, linewidth=2, color='darkviolet', label = u'Ly\u03B1')
            # if DESI_min <= Ly_beta <= DESI_max:
            #     ax3.axvline(Ly_beta, linewidth=2, color='purple', label = u'Ly\u03B2')
            # ax3.set_xlabel('Wavelength / Å')
            # ax3.set_ylabel('Flux / $10^{-17}$ ergs $s^{-1}$ $cm^{-2}$ $Å^{-1}$')
            # ax3.set_title('Gaussian Smoothed Plot of DESI Spectrum')
            # ax3.legend(loc='upper right')

            # fig.subplots_adjust(top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=1.25, wspace=0.2)
            # #top and bottom adjust the vertical space on the top and bottom of the figure.
            # #left and right adjust the horizontal space on the left and right sides.
            # #hspace and wspace adjust the spacing between rows and columns, respectively.

            # # fig.savefig(f'./AGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
            # # fig.savefig(f'./CLAGN Figures/{object_name} - Flux vs Time.png', dpi=300, bbox_inches='tight')
            
        else: 
            #good W1, bad W2
            object_names_list.append(object_name)

            W1_SDSS_interp = np.interp(SDSS_mjd, W1_av_mjd_date, W1_averages_flux)
            W1_DESI_interp = np.interp(DESI_mjd, W1_av_mjd_date, W1_averages_flux)

            W1_SDSS_unc_interp = np.sqrt((((W1_av_mjd_date[after_SDSS_index_W1] - SDSS_mjd)/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[before_SDSS_index_W1])**2 + (((SDSS_mjd - W1_av_mjd_date[before_SDSS_index_W1])/(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1]))*W1_av_uncs_flux[after_SDSS_index_W1])**2)
            W1_DESI_unc_interp = np.sqrt((((W1_av_mjd_date[after_DESI_index_W1] - DESI_mjd)/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[before_DESI_index_W1])**2 + (((DESI_mjd - W1_av_mjd_date[before_DESI_index_W1])/(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1]))*W1_av_uncs_flux[after_DESI_index_W1])**2)

            W1_abs = abs(W1_SDSS_interp-W1_DESI_interp)
            W1_abs_unc = np.sqrt(W1_SDSS_unc_interp**2 + W1_DESI_unc_interp**2)

            W1_av_unc = (1/len(W1_av_uncs_flux))*np.sqrt(sum(unc**2 for unc in W1_av_uncs_flux)) #uncertainty of the mean flux value
            W1_abs_norm = ((W1_abs)/(np.median(W1_averages_flux)))
            W1_abs_norm_unc = W1_abs_norm*np.sqrt(((W1_abs_unc)/(W1_abs))**2 + ((W1_av_unc)/(np.median(W1_averages_flux)))**2)

            W1_z_score_SDSS_DESI = (W1_SDSS_interp-W1_DESI_interp)/(W1_DESI_unc_interp)
            W1_z_score_SDSS_DESI_unc = W1_z_score_SDSS_DESI*((W1_abs_unc)/(W1_abs))
            W1_z_score_DESI_SDSS = (W1_DESI_interp-W1_SDSS_interp)/(W1_SDSS_unc_interp)
            W1_z_score_DESI_SDSS_unc = W1_z_score_DESI_SDSS*((W1_abs_unc)/(W1_abs))

            W1_SDSS_DESI.append(W1_z_score_SDSS_DESI)
            W1_SDSS_DESI_unc.append(W1_z_score_SDSS_DESI_unc)
            W1_DESI_SDSS.append(W1_z_score_DESI_SDSS)
            W1_DESI_SDSS_unc.append(W1_z_score_DESI_SDSS_unc)
            W1_abs_change.append(W1_abs)
            W1_abs_change_unc.append(W1_abs_unc)
            W1_abs_change_norm.append(W1_abs_norm)
            W1_abs_change_norm_unc.append(W1_abs_norm_unc)

            W1_SDSS_bef_dps.append(W1_epoch_dps[before_SDSS_index_W1])
            W1_SDSS_aft_dps.append(W1_epoch_dps[after_SDSS_index_W1])
            W1_DESI_bef_dps.append(W1_epoch_dps[before_DESI_index_W1])
            W1_DESI_aft_dps.append(W1_epoch_dps[after_DESI_index_W1])

            W1_SDSS_gap.append(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1])
            W1_DESI_gap.append(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1])

            W2_abs_norm = np.nan
            W2_abs_norm_unc = np.nan

            W2_z_score_SDSS_DESI = np.nan
            W2_z_score_SDSS_DESI_unc = np.nan
            W2_z_score_DESI_SDSS = np.nan
            W2_z_score_DESI_SDSS_unc = np.nan

            W2_SDSS_DESI.append(np.nan)
            W2_SDSS_DESI_unc.append(np.nan)
            W2_DESI_SDSS.append(np.nan)
            W2_DESI_SDSS_unc.append(np.nan)
            W2_abs_change.append(np.nan)
            W2_abs_change_unc.append(np.nan)
            W2_abs_change_norm.append(np.nan)
            W2_abs_change_norm_unc.append(np.nan)

            W2_SDSS_bef_dps.append(W2_epoch_dps[before_SDSS_index_W2])
            W2_SDSS_aft_dps.append(W2_epoch_dps[after_SDSS_index_W2])
            W2_DESI_bef_dps.append(W2_epoch_dps[before_DESI_index_W2])
            W2_DESI_aft_dps.append(W2_epoch_dps[after_DESI_index_W2])

            W2_SDSS_gap.append(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2])
            W2_DESI_gap.append(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2])

            zscores = np.sort([W1_z_score_SDSS_DESI, W1_z_score_DESI_SDSS, W2_z_score_SDSS_DESI, W2_z_score_DESI_SDSS]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_SDSS_DESI_unc, W1_z_score_DESI_SDSS_unc, W2_z_score_SDSS_DESI_unc, W2_z_score_DESI_SDSS_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(abs(zscores)))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(abs(zscores)))
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

            if SDSS_min < 3000 and SDSS_max > 3920 and DESI_min < 3000 and DESI_max > 3920:
                closest_index_lower_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
                closest_index_upper_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
                sdss_blue_lamb = sdss_lamb[closest_index_lower_sdss:closest_index_upper_sdss]
                sdss_blue_flux = sdss_flux[closest_index_lower_sdss:closest_index_upper_sdss]
                sdss_blue_flux_smooth = Gaus_smoothed_SDSS[closest_index_lower_sdss:closest_index_upper_sdss]

                desi_lamb = desi_lamb.tolist()
                closest_index_lower_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
                closest_index_upper_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
                desi_blue_lamb = desi_lamb[closest_index_lower_desi:closest_index_upper_desi]
                desi_blue_flux = desi_flux[closest_index_lower_desi:closest_index_upper_desi]
                desi_blue_flux_smooth = Gaus_smoothed_DESI[closest_index_lower_desi:closest_index_upper_desi]

                #interpolating SDSS flux so lambda values match up with DESI . Done this way round because DESI lambda values are closer together.
                sdss_interp_fn = interp1d(sdss_blue_lamb, sdss_blue_flux_smooth, kind='linear', fill_value='extrapolate')
                sdss_blue_flux_interp = sdss_interp_fn(desi_blue_lamb) #interpolating the sdss flux to be in line with the desi lambda values

                UV_flux_change = [desi - sdss for sdss, desi in zip(sdss_blue_flux_interp, desi_blue_flux_smooth)]

                mean_UV_flux_change.append(np.mean(UV_flux_change))
                mean_UV_flux_change_unc.append(np.std(UV_flux_change))
            else:
                mean_UV_flux_change.append(np.nan)
                mean_UV_flux_change_unc.append(np.nan)

    else: #Bad W1
        if w == 0 and r == 0: #Good W2 if true
            #Bad W1, good W2
            object_names_list.append(object_name)

            W1_abs_norm = np.nan
            W1_abs_norm_unc = np.nan

            W1_z_score_SDSS_DESI = np.nan
            W1_z_score_SDSS_DESI_unc = np.nan
            W1_z_score_DESI_SDSS = np.nan
            W1_z_score_DESI_SDSS_unc = np.nan

            W1_SDSS_DESI.append(np.nan)
            W1_SDSS_DESI_unc.append(np.nan)
            W1_DESI_SDSS.append(np.nan)
            W1_DESI_SDSS_unc.append(np.nan)
            W1_abs_change.append(np.nan)
            W1_abs_change_unc.append(np.nan)
            W1_abs_change_norm.append(np.nan)
            W1_abs_change_norm_unc.append(np.nan)

            W1_SDSS_bef_dps.append(W1_epoch_dps[before_SDSS_index_W1])
            W1_SDSS_aft_dps.append(W1_epoch_dps[after_SDSS_index_W1])
            W1_DESI_bef_dps.append(W1_epoch_dps[before_DESI_index_W1])
            W1_DESI_aft_dps.append(W1_epoch_dps[after_DESI_index_W1])

            W1_SDSS_gap.append(W1_av_mjd_date[after_SDSS_index_W1] - W1_av_mjd_date[before_SDSS_index_W1])
            W1_DESI_gap.append(W1_av_mjd_date[after_DESI_index_W1] - W1_av_mjd_date[before_DESI_index_W1])

            W2_SDSS_interp = np.interp(SDSS_mjd, W2_av_mjd_date, W2_averages_flux)
            W2_DESI_interp = np.interp(DESI_mjd, W2_av_mjd_date, W2_averages_flux)

            W2_SDSS_unc_interp = np.sqrt((((W2_av_mjd_date[after_SDSS_index_W2] - SDSS_mjd)/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[before_SDSS_index_W2])**2 + (((SDSS_mjd - W2_av_mjd_date[before_SDSS_index_W2])/(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2]))*W2_av_uncs_flux[after_SDSS_index_W2])**2)
            W2_DESI_unc_interp = np.sqrt((((W2_av_mjd_date[after_DESI_index_W2] - DESI_mjd)/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[before_DESI_index_W2])**2 + (((DESI_mjd - W2_av_mjd_date[before_DESI_index_W2])/(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2]))*W2_av_uncs_flux[after_DESI_index_W2])**2)

            W2_abs = abs(W2_SDSS_interp-W2_DESI_interp)
            W2_abs_unc = np.sqrt(W2_SDSS_unc_interp**2 + W2_DESI_unc_interp**2)

            W2_av_unc = (1/len(W2_av_uncs_flux))*np.sqrt(sum(unc**2 for unc in W2_av_uncs_flux)) #uncertainty of the mean flux value
            W2_abs_norm = ((W2_abs)/(np.median(W2_averages_flux)))
            W2_abs_norm_unc = W2_abs_norm*np.sqrt(((W2_abs_unc)/(W2_abs))**2 + ((W2_av_unc)/(np.median(W2_averages_flux)))**2)

            W2_z_score_SDSS_DESI = (W2_SDSS_interp-W2_DESI_interp)/(W2_DESI_unc_interp)
            W2_z_score_SDSS_DESI_unc = W2_z_score_SDSS_DESI*((W2_abs_unc)/(W2_abs))
            W2_z_score_DESI_SDSS = (W2_DESI_interp-W2_SDSS_interp)/(W2_SDSS_unc_interp)
            W2_z_score_DESI_SDSS_unc = W2_z_score_DESI_SDSS*((W2_abs_unc)/(W2_abs))

            W2_SDSS_DESI.append(W2_z_score_SDSS_DESI)
            W2_SDSS_DESI_unc.append(W2_z_score_SDSS_DESI_unc)
            W2_DESI_SDSS.append(W2_z_score_DESI_SDSS)
            W2_DESI_SDSS_unc.append(W2_z_score_DESI_SDSS_unc)
            W2_abs_change.append(W2_abs)
            W2_abs_change_unc.append(W2_abs_unc)
            W2_abs_change_norm.append(W2_abs_norm)
            W2_abs_change_norm_unc.append(W2_abs_norm_unc)

            W2_SDSS_bef_dps.append(W2_epoch_dps[before_SDSS_index_W2])
            W2_SDSS_aft_dps.append(W2_epoch_dps[after_SDSS_index_W2])
            W2_DESI_bef_dps.append(W2_epoch_dps[before_DESI_index_W2])
            W2_DESI_aft_dps.append(W2_epoch_dps[after_DESI_index_W2])

            W2_SDSS_gap.append(W2_av_mjd_date[after_SDSS_index_W2] - W2_av_mjd_date[before_SDSS_index_W2])
            W2_DESI_gap.append(W2_av_mjd_date[after_DESI_index_W2] - W2_av_mjd_date[before_DESI_index_W2])

            zscores = np.sort([W2_z_score_SDSS_DESI, W1_z_score_DESI_SDSS, W2_z_score_SDSS_DESI, W2_z_score_DESI_SDSS]) #sorts in ascending order, nans at end
            zscore_uncs = np.sort([W1_z_score_SDSS_DESI_unc, W1_z_score_DESI_SDSS_unc, W2_z_score_SDSS_DESI_unc, W2_z_score_DESI_SDSS_unc])
            if np.isnan(zscores[0]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be nan - all values are nan
                mean_zscore_unc.append(np.nan)
            elif np.isnan(zscores[1]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be zscores[0] - only non nan value
                mean_zscore_unc.append(abs(zscore_uncs[0]))
            elif np.isnan(zscores[2]) == True:
                mean_zscore.append(np.nanmean(abs(zscores))) #will be 1/2(zscores[0]+zscores[1])
                mean_zscore_unc.append((1/2)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2))
            elif np.isnan(zscores[3]) == True:
                mean_zscore.append(np.nanmean(abs(zscores)))
                mean_zscore_unc.append((1/3)*np.sqrt(zscore_uncs[0]**2+zscore_uncs[1]**2 + zscore_uncs[2]**2))
            else:
                mean_zscore.append(np.nanmean(abs(zscores)))
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

            if SDSS_min < 3000 and SDSS_max > 3920 and DESI_min < 3000 and DESI_max > 3920:
                closest_index_lower_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
                closest_index_upper_sdss = min(range(len(sdss_lamb)), key=lambda i: abs(sdss_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
                sdss_blue_lamb = sdss_lamb[closest_index_lower_sdss:closest_index_upper_sdss]
                sdss_blue_flux = sdss_flux[closest_index_lower_sdss:closest_index_upper_sdss]
                sdss_blue_flux_smooth = Gaus_smoothed_SDSS[closest_index_lower_sdss:closest_index_upper_sdss]

                desi_lamb = desi_lamb.tolist()
                closest_index_lower_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3000)) #3000 to avoid Mg2 emission line
                closest_index_upper_desi = min(range(len(desi_lamb)), key=lambda i: abs(desi_lamb[i] - 3920)) #3920 to avoid K Fraunhofer line
                desi_blue_lamb = desi_lamb[closest_index_lower_desi:closest_index_upper_desi]
                desi_blue_flux = desi_flux[closest_index_lower_desi:closest_index_upper_desi]
                desi_blue_flux_smooth = Gaus_smoothed_DESI[closest_index_lower_desi:closest_index_upper_desi]

                #interpolating SDSS flux so lambda values match up with DESI . Done this way round because DESI lambda values are closer together.
                sdss_interp_fn = interp1d(sdss_blue_lamb, sdss_blue_flux_smooth, kind='linear', fill_value='extrapolate')
                sdss_blue_flux_interp = sdss_interp_fn(desi_blue_lamb) #interpolating the sdss flux to be in line with the desi lambda values

                UV_flux_change = [desi - sdss for sdss, desi in zip(sdss_blue_flux_interp, desi_blue_flux_smooth)]

                mean_UV_flux_change.append(np.mean(UV_flux_change))
                mean_UV_flux_change_unc.append(np.std(UV_flux_change))
            else:
                mean_UV_flux_change.append(np.nan)
                mean_UV_flux_change_unc.append(np.nan)
  
        else:
            #bad W1, bad W2
            print('Bad W1 & W2 data')
            continue

#for loop now ended
quantifying_change_data = {
    "Object": object_names_list, #0

    "W1 Z Score SDSS vs DESI": W1_SDSS_DESI, #1
    "W1 Z Score SDSS vs DESI Unc": W1_SDSS_DESI_unc, #2
    "W1 Z Score DESI vs SDSS": W1_DESI_SDSS, #3
    "W1 Z Score DESI vs SDSS Unc": W1_DESI_SDSS_unc, #4
    "W1 Flux Change": W1_abs_change, #5
    "W1 Flux Change Unc": W1_abs_change_unc, #6
    "W1 Normalised Flux Change": W1_abs_change_norm, #7
    "W1 Normalised Flux Change Unc": W1_abs_change_norm_unc, #8

    "W2 Z Score SDSS vs DESI": W2_SDSS_DESI, #9
    "W2 Z Score SDSS vs DESI Unc": W2_SDSS_DESI_unc, #10
    "W2 Z Score DESI vs SDSS": W2_DESI_SDSS, #11
    "W2 Z Score DESI vs SDSS Unc": W2_DESI_SDSS_unc, #12
    "W2 Flux Change": W2_abs_change, #13
    "W2 Flux Change Unc": W2_abs_change_unc, #14
    "W2 Normalised Flux Change": W2_abs_change_norm, #15
    "W2 Normalised Flux Change Unc": W2_abs_change_norm_unc, #16

    "W1 before SDSS Epoch DPs": W1_SDSS_bef_dps, #17
    "W1 after SDSS Epoch DPs": W1_SDSS_aft_dps, #18
    "W1 before DESI Epoch DPs": W1_DESI_bef_dps, #19
    "W1 after DESI Epoch DPs": W1_DESI_aft_dps, #20
    "W2 before SDSS Epoch DPs": W2_SDSS_bef_dps, #21
    "W2 after SDSS Epoch DPs": W2_SDSS_aft_dps, #22
    "W2 before DESI Epoch DPs": W2_DESI_bef_dps, #23
    "W2 after DESI Epoch DPs": W2_DESI_aft_dps, #24

    "Mean Z Score": mean_zscore, #25
    "Mean Z Score Unc": mean_zscore_unc, #26. # Note that this is not the median uncertainty. It is the uncertainty in the median z score reading. 
    "Mean Normalised Flux Change": mean_norm_flux_change, #27
    "Mean Normalised Flux Change Unc": mean_norm_flux_change_unc, #28

    "Mean UV Flux Change DESI - SDSS": mean_UV_flux_change, #29
    "Mean UV Flux Change DESI - SDSS Unc": mean_UV_flux_change_unc, #30

    "W1 SDSS Gap": W1_SDSS_gap, #31
    "W1 DESI Gap": W1_DESI_gap, #32
    "W2 SDSS Gap": W2_SDSS_gap, #33
    "W2 DESI Gap": W2_DESI_gap, #34
}

# Convert the data into a DataFrame
df = pd.DataFrame(quantifying_change_data)

#Creating a csv file of my data
df.to_csv("CLAGN_Quantifying_Change.csv", index=False)
# df.to_csv("AGN_Quantifying_Change.csv", index=False)