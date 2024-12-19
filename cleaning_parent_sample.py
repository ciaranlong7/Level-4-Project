import numpy as np
import pandas as pd
from sparcl.client import SparclClient
from requests.exceptions import ConnectTimeout
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

client = SparclClient(connect_timeout=10)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type((ConnectTimeout, TimeoutError, ConnectionError)))
def get_primary_spectrum(specid): #some objects have multiple spectra for it in DESI- the best one is the 'primary' spectrum    
    try:
        res = client.retrieve_by_specid(specid_list=[int(specid)], include=['specprimary'], dataset_list=['DESI-EDR'])

        records = res.records

        if not records: #no spectrum could be found:
            return 0

        spec_primary = np.array([records[jj].specprimary for jj in range(len(records))])

        if not np.any(spec_primary): #no primary spectrum could be found
            return 0
        
        return 1

    except (ConnectTimeout, TimeoutError, ConnectionError) as e:
        temp_save = guo_parent[guo_parent['keep'] == 1]

        temp_save = temp_save.drop(columns=['keep'])

        temp_save.to_csv('temp_sample.csv', index=False)

        print(f"Connection timeout: {e}")
        print('Temporary save to temp_sample.csv successful')
        raise ConnectTimeout
    
# Step 1: Read the CSV file
guo_parent = pd.read_csv('guo23_parent_sample_no_duplicates.csv')

# guo_parent = guo_parent.iloc[:30000, :] #checking first 30k rows
# guo_parent = guo_parent.iloc[30000:55000, :] #checking rows 30k-55k
guo_parent = guo_parent.iloc[55000:, :] #checking rows 55k until finish

# Step 2: Apply the function to the 3rd column (index 2)
guo_parent['keep'] = guo_parent.iloc[:, 10].apply(get_primary_spectrum)

# Step 3: Filter rows where the function returns 1
new_parent = guo_parent[guo_parent['keep'] == 1]

# Step 4: Remove the 'keep' column
new_parent = new_parent.drop(columns=['keep'])

# Step 5: Write the filtered DataFrame to a new CSV file
# new_parent.to_csv('new_parent_sample.csv', index=False)
# new_parent.to_csv('new_parent_sample_30k.csv', index=False)
new_parent.to_csv('new_parent_sample_55k.csv', index=False)


# ## Combining the three data frames created
# guo_parent = pd.read_csv('new_parent_sample.csv')
# guo_parent_thirtyk = pd.read_csv('new_parent_sample_30k.csv')
# guo_parent_fivefivek = pd.read_csv('new_parent_sample_55k.csv')
# combined_df = pd.concat([guo_parent, guo_parent_thirtyk, guo_parent_fivefivek], ignore_index=True)
# combined_df.to_csv('combined_guo_parent_sample.csv', index=False)

# parent_sample = pd.read_csv('combined_guo_parent_sample.csv')
# print(f'Objects in parent sample, before duplicates removed = {len(parent_sample)}')
# columns_to_check = parent_sample.columns[[3, 10]] #checking SDSS name, DESI name
# parent_sample = parent_sample.drop_duplicates(subset=columns_to_check)
# print(f'Objects in parent sample, after duplicates removed = {len(parent_sample)}')


# ## Checking redshift
# parent_sample = pd.read_csv('combined_guo_parent_sample.csv')
# same_redshift = parent_sample[np.abs(parent_sample.iloc[:, 2] - parent_sample.iloc[:, 9]) <= 0.01]
# different_redshift = parent_sample[np.abs(parent_sample.iloc[:, 2] - parent_sample.iloc[:, 9]) > 0.01]

# print(f'Objects in parent sample with same redshift for SDSS & DESI = {len(same_redshift)}')
# print(f'Objects in parent sample with different redshift for SDSS & DESI = {len(different_redshift)}')

# columns_to_check = parent_sample.columns[[3, 10]] #checking SDSS name, DESI name
# same_redshift = same_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects in cleaned sample after duplicates removed = {len(same_redshift)}')

# same_redshift.to_csv('clean_parent_sample.csv', index=False)
# different_redshift.to_csv('outside_redshift_sample.csv', index=False)


# ##Checking if any Guo CLAGN are in sample
# Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")

# object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

# no_CLAGN = same_redshift[~same_redshift.iloc[:, 3].isin(object_names)]
# print(f'Objects in cleaned sample after CLAGN removed = {len(no_CLAGN)}')

# no_CLAGN.to_csv('clean_parent_sample_no_CLAGN.csv', index=False)