import numpy as np
import pandas as pd
from sparcl.client import SparclClient
from requests.exceptions import ConnectTimeout
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

client = SparclClient(connect_timeout=10)
inc = client.get_all_fields()

@retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(ConnectionError))
def get_primary_spectrum(specid): #some objects have multiple spectra for it in DESI- the best one is the 'primary' spectrum    
    try:
        res = client.retrieve_by_specid(specid_list=[int(specid)], include=inc, dataset_list=['DESI-EDR'])

        records = res.records

        if not records: #no spectrum could be found:
            return 0

        spec_primary = np.array([records[jj].specprimary for jj in range(len(records))])

        if not np.any(spec_primary): #no primary spectrum could be found
            return 0
        
        return 1

    except ConnectTimeout as e:
            temp_save = guo_parent[guo_parent['keep'] == 1]

            temp_save = temp_save.drop(columns=['keep'])

            temp_save.to_csv('temp_sample.csv', index=False)

            print(f"Connection timeout: {e}")
            print('Temporary save to temp_sample.csv successful')
            raise ConnectTimeout
    
# Step 1: Read the CSV file
guo_parent = pd.read_csv('guo23_parent_sample_no_duplicates.csv')

# Step 2: Apply the function to the 3rd column (index 2)
guo_parent['keep'] = guo_parent.iloc[:, 10].apply(get_primary_spectrum)

# Step 3: Filter rows where the function returns 1
new_parent = guo_parent[guo_parent['keep'] == 1]

# Step 4: Remove the 'keep' column
new_parent = new_parent.drop(columns=['keep'])

# Step 5: Write the filtered DataFrame to a new CSV file
new_parent.to_csv('new_parent_sample.csv', index=False)