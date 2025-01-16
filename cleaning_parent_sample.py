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
    
guo_parent = pd.read_csv('guo23_parent_sample_no_duplicates.csv')

# # guo_parent = guo_parent.iloc[:30000, :] #checking first 30k rows
# # guo_parent = guo_parent.iloc[30000:55000, :] #checking rows 30k-55k
# guo_parent = guo_parent.iloc[55000:, :] #checking rows 55k until finish

# # Step 2: Apply the function to the 3rd column (index 2)
# guo_parent['keep'] = guo_parent.iloc[:, 10].apply(get_primary_spectrum)

# # Step 3: Filter rows where the function returns 1
# new_parent = guo_parent[guo_parent['keep'] == 1]

# # Step 4: Remove the 'keep' column
# new_parent = new_parent.drop(columns=['keep'])

# # Step 5: Write the filtered DataFrame to a new CSV file
# # new_parent.to_csv('new_parent_sample.csv', index=False)
# # new_parent.to_csv('new_parent_sample_30k.csv', index=False)
# new_parent.to_csv('new_parent_sample_55k.csv', index=False)


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


# # Checking if any instances of an object having SDSS & DESI redshift being > 0.01 apart actually had a redshift that lined up,
# # but was originally removed in the original removal of duplicates - cutting from 110k to 80k
# outside_redshift = pd.read_csv('outside_redshift_sample.csv')
# print(f'number of objects with different redshift = {len(outside_redshift)}')
# guo_parent = pd.read_csv('guo23_parent_sample.csv')

# names_outside_redshift = outside_redshift.iloc[:, 3]

# # Find all rows in guo_parent where the name (4th column) matches any name in the 4th column of CSV A
# matching_rows = guo_parent[guo_parent.iloc[:, 4].isin(names_outside_redshift)]
# print(f'number of matching rows = {len(matching_rows)}')

# same_redshift = matching_rows[np.abs(matching_rows.iloc[:, 3] - matching_rows.iloc[:, 10]) <= 0.01]
# different_redshift = matching_rows[np.abs(matching_rows.iloc[:, 3] - matching_rows.iloc[:, 10]) > 0.01]

# print(f'Objects recovered with a matching redshift that originally was removed (with duplicates) = {len(same_redshift)}')
# print(f'Objects in matching_rows with different redshift for SDSS & DESI (with duplicates) = {len(different_redshift)}')

# columns_to_check = matching_rows.columns[[4, 11]] #checking SDSS name, DESI name
# same_redshift = same_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects recovered with a matching redshift after duplicates removed = {len(same_redshift)}')

# same_redshift = same_redshift.drop(same_redshift.columns[0], axis=1) #get rid of index column

# same_redshift_file = 'clean_parent_sample.csv'

# same_redshift.to_csv(same_redshift_file, mode='a', index=False, header=False) #mode a = append mode

# different_redshift = different_redshift.drop_duplicates(subset=columns_to_check)
# print(f'Objects still with a different redshift after duplicates removed = {len(different_redshift)}')
# different_redshift.to_csv('outside_redshift_sample.csv', index=False) 
# #there now are some instances of an object having its different redshift instance in the different csv
# #and it also has a instance of same redshift in the same redshift csv. But doesn't really matter


# clean_sample = pd.read_csv('clean_parent_sample.csv') #sample after recovered objects are appended.
# print(f'length of clean sample = {len(clean_sample)}')
# columns_to_check = clean_sample.columns[[3, 10]] #checking SDSS name, DESI name
# clean_sample = clean_sample.drop_duplicates(subset=columns_to_check)
# clean_sample.to_csv('clean_parent_sample.csv', index=False) 
# print(f'length of clean sample after dropping duplicates = {len(clean_sample)}')
# #This recovered 6199 objects.


# ##Checking if any Guo CLAGN are in sample
# Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")
# clean_sample = pd.read_csv('clean_parent_sample.csv')

# object_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]

# no_CLAGN = clean_sample[~clean_sample.iloc[:, 3].isin(object_names)]
# print(f'Objects in cleaned sample after CLAGN removed = {len(no_CLAGN)}')

# no_CLAGN.to_csv('clean_parent_sample_no_CLAGN.csv', index=False)