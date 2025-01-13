import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

parent_sample = pd.read_csv('clean_parent_sample_no_CLAGN.csv')
Guo_table4 = pd.read_csv("Guo23_table4_clagn.csv")

CLAGN_names = [object_name for object_name in Guo_table4.iloc[:, 0] if pd.notna(object_name)]
AGN_names = parent_sample.iloc[:, 3].sample(n=250, random_state=42)

CLAGN_redshifts = []
for object_name in CLAGN_names:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    CLAGN_redshifts.append(redshift)

AGN_redshifts = []
for object_name in AGN_names:
    object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
    redshift = object_data.iloc[0, 2]
    AGN_redshifts.append(redshift)

median_CLAGN_redshift = np.nanmedian(CLAGN_redshifts)
median_AGN_redshift = np.nanmedian(AGN_redshifts)
print(f'Median CLAGN sample redshift = {median_CLAGN_redshift:.3f}')
print(f'Median AGN sample redshift = {median_AGN_redshift:.3f}')

#Quantifying change data
CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_2nd_biggest_smallest.csv')
# CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_2nd_biggest_smallest_mean_uncs.csv')
print(f'Number of CLAGN: {len(CLAGN_quantifying_change_data)}')
CLAGN_zscores = CLAGN_quantifying_change_data.iloc[:, 17].tolist()  # 18th column
CLAGN_zscore_uncs = CLAGN_quantifying_change_data.iloc[:, 18].tolist()
CLAGN_norm_flux_change = CLAGN_quantifying_change_data.iloc[:, 19].tolist()
CLAGN_norm_flux_change_unc = CLAGN_quantifying_change_data.iloc[:, 20].tolist()
CLAGN_W1_low_flux = CLAGN_quantifying_change_data.iloc[:, 25].tolist()
CLAGN_W1_median_dev_flux = CLAGN_quantifying_change_data.iloc[:, 27].tolist() #median_abs_dev of flux for an object in w1 band
CLAGN_W1_median_flux_unc = CLAGN_quantifying_change_data.iloc[:, 28].tolist() #median of uncs for an object's flux in W1 band
CLAGN_W1_median_dev_flux_unc = CLAGN_quantifying_change_data.iloc[:, 29].tolist() #median_abs_dev of uncs for an object's flux in W1 band
CLAGN_W2_low_flux = CLAGN_quantifying_change_data.iloc[:, 26].tolist()
CLAGN_W2_median_dev_flux = CLAGN_quantifying_change_data.iloc[:, 30].tolist()
CLAGN_W2_median_flux_unc = CLAGN_quantifying_change_data.iloc[:, 31].tolist()
CLAGN_W2_median_dev_flux_unc = CLAGN_quantifying_change_data.iloc[:, 32].tolist()
CLAGN_names_analysis = CLAGN_quantifying_change_data.iloc[:, 0].tolist()

CLAGN_new_norm_change = []
for i in range(len(CLAGN_quantifying_change_data)):
    CLAGN_new_norm_change.append(CLAGN_norm_flux_change[i]/CLAGN_W1_median_flux_unc[i])

AGN_quantifying_change_data = pd.read_csv('AGN_Quantifying_Change_just_MIR_2nd_biggest_smallest.csv')
# AGN_quantifying_change_data = pd.read_csv('AGN_Quantifying_Change_just_MIR_2nd_biggest_smallest_mean_uncs.csv')
print(f'Number of AGN: {len(AGN_quantifying_change_data)}')
AGN_zscores = AGN_quantifying_change_data.iloc[:, 17].tolist()
AGN_zscore_uncs = AGN_quantifying_change_data.iloc[:, 18].tolist()
AGN_norm_flux_change = AGN_quantifying_change_data.iloc[:, 19].tolist()
AGN_norm_flux_change_unc = AGN_quantifying_change_data.iloc[:, 20].tolist()
AGN_W1_low_flux = AGN_quantifying_change_data.iloc[:, 25].tolist()
AGN_W1_median_dev_flux = AGN_quantifying_change_data.iloc[:, 27].tolist() #median_abs_dev of flux for an object in w1 band
AGN_W1_median_flux_unc = AGN_quantifying_change_data.iloc[:, 28].tolist() #median of uncs for an object's flux in W1 band
AGN_W1_median_dev_flux_unc = AGN_quantifying_change_data.iloc[:, 29].tolist() #median_abs_dev of uncs for an object's flux in W1 band
AGN_W2_low_flux = AGN_quantifying_change_data.iloc[:, 26].tolist()
AGN_W2_median_dev_flux = AGN_quantifying_change_data.iloc[:, 30].tolist()
AGN_W2_median_flux_unc = AGN_quantifying_change_data.iloc[:, 31].tolist()
AGN_W2_median_dev_flux_unc = AGN_quantifying_change_data.iloc[:, 32].tolist()
AGN_names_analysis = AGN_quantifying_change_data.iloc[:, 0].tolist()

AGN_new_norm_change = []
for i in range(len(AGN_quantifying_change_data)):
    AGN_new_norm_change.append(AGN_norm_flux_change[i]/AGN_W1_median_flux_unc[i])

print(f'CLAGN W1 median 2nd lowest flux = {np.nanmedian(CLAGN_W1_low_flux):.4f}')
print(f'CLAGN W1 median median_abs_dev flux = {np.nanmedian(CLAGN_W1_median_dev_flux):.5f}')
print(f'CLAGN W1 median uncertainty in flux = {np.nanmedian(CLAGN_W1_median_flux_unc):.5f}')
print(f'CLAGN W1 median median_abs_deviation uncertainties in flux = {np.nanmedian(CLAGN_W1_median_dev_flux_unc):.4f}')
print(f'CLAGN W2 median 2nd lowest flux = {np.nanmedian(CLAGN_W2_low_flux):.4f}')
print(f'CLAGN W2 median median_abs_dev flux = {np.nanmedian(CLAGN_W2_median_dev_flux):.5f}')
print(f'CLAGN W2 median uncertainty in flux = {np.nanmedian(CLAGN_W2_median_flux_unc):.5f}')
print(f'CLAGN W2 median median_abs_deviation uncertainties in flux = {np.nanmedian(CLAGN_W2_median_dev_flux_unc):.4f}')

print(f'AGN W1 median 2nd lowest flux = {np.nanmedian(AGN_W1_low_flux):.4f}')
print(f'AGN W1 median median_abs_dev flux = {np.nanmedian(AGN_W1_median_dev_flux):.5f}')
print(f'AGN W1 median uncertainty in flux = {np.nanmedian(AGN_W1_median_flux_unc):.5f}')
print(f'AGN W1 median median_abs_deviation uncertainties in flux = {np.nanmedian(AGN_W1_median_dev_flux_unc):.4f}')
print(f'AGN W2 median 2nd lowest flux = {np.nanmedian(AGN_W2_low_flux):.4f}')
print(f'AGN W2 median median_abs_dev flux = {np.nanmedian(AGN_W2_median_dev_flux):.5f}')
print(f'AGN W2 median uncertainty in flux = {np.nanmedian(AGN_W2_median_flux_unc):.5f}')
print(f'AGN W2 median median_abs_deviation uncertainties in flux = {np.nanmedian(AGN_W2_median_dev_flux_unc):.4f}')

CLAGN_redshifts = []
for object_name in CLAGN_names_analysis:
    object_row = Guo_table4[Guo_table4.iloc[:, 0] == object_name]
    redshift = object_row.iloc[0, 3]
    CLAGN_redshifts.append(redshift)

AGN_redshifts = []
for object_name in AGN_names_analysis:
    object_data = parent_sample[parent_sample.iloc[:, 3] == object_name]
    redshift = object_data.iloc[0, 2]
    AGN_redshifts.append(redshift)

median_CLAGN_redshift = np.nanmedian(CLAGN_redshifts)
median_AGN_redshift = np.nanmedian(AGN_redshifts)
print(f'Median CLAGN analysed redshift = {median_CLAGN_redshift:.3f}')
print(f'Median AGN analysed redshift = {median_AGN_redshift:.3f}')

#want the median value of the random sample of AGN
median_norm_flux_change = np.nanmedian(AGN_norm_flux_change)
median_norm_flux_change_unc = np.nanmedian(AGN_norm_flux_change_unc)
three_sigma_norm_flux_change = median_norm_flux_change + 3*median_norm_flux_change_unc
median_zscore = np.nanmedian(AGN_zscores)
median_zscore_unc = np.nanmedian(AGN_zscore_uncs)
three_sigma_zscore = median_zscore + 3*median_zscore_unc
print(f'Median norm flux difference = {median_norm_flux_change:.4f}')
print(f'Median norm flux difference unc = {median_norm_flux_change_unc:.4f}')
print(f'3\u03C3 significance for norm flux difference = {three_sigma_norm_flux_change:.4f}')
print(f'Median z score = {median_zscore:.4f}')
print(f'Median z score unc = {median_zscore_unc:.4f}')
print(f'3\u03C3 significance for z score = {three_sigma_zscore:.4f}')

median_norm_flux_change_CLAGN = np.nanmedian(CLAGN_norm_flux_change)
median_zscore_CLAGN = np.nanmedian(CLAGN_zscores)

i = 0
for zscore in CLAGN_zscores:
    if zscore > three_sigma_zscore:
        i += 1

j = 0
for zscore in AGN_zscores:
    if zscore > three_sigma_zscore:
        j += 1

k = 0
for normchange in CLAGN_norm_flux_change:
    if normchange > three_sigma_norm_flux_change:
        k += 1

l = 0
for normchange in AGN_norm_flux_change:
    if normchange > three_sigma_norm_flux_change:
        l += 1

print(f'{i}/{len(CLAGN_zscores)}={i/len(CLAGN_zscores)*100:.3f}% of CLAGN above zscore threshold')
print(f'{k}/{len(CLAGN_norm_flux_change)}={k/len(CLAGN_norm_flux_change)*100:.3f}% of CLAGN above norm_change threshold')
print(f'{j}/{len(AGN_zscores)}={j/len(AGN_zscores)*100:.3f}% of AGN above zscore threshold')
print(f'{l}/{len(AGN_norm_flux_change)}={l/len(AGN_norm_flux_change)*100:.3f}% of AGN above norm_change threshold')

### BELOW INVESTIGATION CHECKS WHETHER ELIMINATING OBJECTS WITH A HIGH UNC RATIO (eg NFD_UNC/NFD) IMPROVES RESULTS.
### I find that eliminating objects with a high unc ratio increases the amount of CLAGN & non-CL AGN that are above the thresholds.
### However, this is in a proportionate manner. 
### I.e., ratio of CLAGN to non CL AGN above the thresholds remains ~10x for NFD and ~3x for zscore.

CLAGN_norm_uncpc = []
CLAGN_z_uncpc = []
test_CLAGN_zscores = []
test_CLAGN_nfd = []
for i in range(len(CLAGN_quantifying_change_data)):
    CLAGN_norm_uncpc.append(CLAGN_norm_flux_change_unc[i]/CLAGN_norm_flux_change[i])
    CLAGN_z_uncpc.append(CLAGN_zscore_uncs[i]/CLAGN_zscores[i])
    if CLAGN_zscore_uncs[i]/CLAGN_zscores[i] > 0.5:
        # print(f'CLAGN Unc % = {CLAGN_norm_flux_change_unc[i]/CLAGN_norm_flux_change[i]*100:.3f}')
        # print(f'CLAGN Z Score = {CLAGN_zscores[i]}')
        # print(f'CLAGN NFD = {CLAGN_norm_flux_change[i]}')
        continue
    else: #unc % <50
        test_CLAGN_zscores.append(CLAGN_zscores[i])
        test_CLAGN_nfd.append(CLAGN_norm_flux_change[i])

CLAGN_norm_fifty = len([x for x in CLAGN_norm_uncpc if x > 0.5])
CLAGN_z_fifty = len([x for x in CLAGN_z_uncpc if x > 0.5])

print(f'{CLAGN_norm_fifty/len(CLAGN_quantifying_change_data)*100:.3f}% of CLAGN have a NFD unc > 50% of NFD value')
print(f'{CLAGN_z_fifty/len(CLAGN_quantifying_change_data)*100:.3f}% of CLAGN have a zscore unc > 50% of zscore value')

AGN_norm_uncpc = []
AGN_z_uncpc = []
test_AGN_zscores = []
test_AGN_zscore_unc = []
test_AGN_nfd = []
test_AGN_nfd_unc = []
for i in range(len(AGN_quantifying_change_data)):
    AGN_norm_uncpc.append(AGN_norm_flux_change_unc[i]/AGN_norm_flux_change[i])
    AGN_z_uncpc.append(AGN_zscore_uncs[i]/AGN_zscores[i])
    if AGN_zscore_uncs[i]/AGN_zscores[i] > 0.5:
        # print(f'AGN Unc % = {AGN_norm_flux_change_unc[i]/AGN_norm_flux_change[i]*100:.3f}')
        # print(f'AGN Z Score = {AGN_zscores[i]}')
        # print(f'AGN NFD = {AGN_norm_flux_change[i]}')
        continue
    else: #unc % <50
        test_AGN_zscores.append(AGN_zscores[i])
        test_AGN_zscore_unc.append(AGN_zscore_uncs[i])
        test_AGN_nfd.append(AGN_norm_flux_change[i])
        test_AGN_nfd_unc.append(AGN_norm_flux_change_unc[i])

AGN_norm_fifty = len([x for x in AGN_norm_uncpc if x > 0.5])
AGN_z_fifty = len([x for x in CLAGN_z_uncpc if x > 0.5])

print(f'{AGN_norm_fifty/len(AGN_quantifying_change_data)*100:.3f}% of AGN have a NFD unc > 50% of NFD value')
print(f'{AGN_z_fifty/len(AGN_quantifying_change_data)*100:.3f}% of AGN have a zscore unc > 50% of zscore value')

test_median_norm_flux_change = np.nanmedian(test_AGN_nfd)
test_median_norm_flux_change_unc = np.nanmedian(test_AGN_nfd_unc)
test_three_sigma_norm_flux_change = test_median_norm_flux_change + 3*test_median_norm_flux_change_unc
test_median_zscore = np.nanmedian(test_AGN_zscores)
test_median_zscore_unc = np.nanmedian(test_AGN_zscore_unc)
test_three_sigma_zscore = test_median_zscore + 3*test_median_zscore_unc

print(f'Test Median norm flux difference = {test_median_norm_flux_change:.4f}')
print(f'Test Median norm flux difference unc = {test_median_norm_flux_change_unc:.4f}')
print(f'Test 3\u03C3 significance for norm flux difference = {test_three_sigma_norm_flux_change:.4f}')
print(f'Test Median z score = {test_median_zscore:.4f}')
print(f'Test Median z score unc = {test_median_zscore_unc:.4f}')
print(f'Test 3\u03C3 significance for z score = {test_three_sigma_zscore:.4f}')

i = 0
for zscore in test_CLAGN_zscores:
    if zscore > test_three_sigma_zscore:
        i += 1
j = 0
for zscore in test_AGN_zscores:
    if zscore > test_three_sigma_zscore:
        j += 1
k = 0
for normchange in test_CLAGN_nfd:
    if normchange > test_three_sigma_norm_flux_change:
        k += 1
l = 0
for normchange in test_AGN_nfd:
    if normchange > test_three_sigma_norm_flux_change:
        l += 1

print('Eliminated Objects with unc % > 50:')
print(f'{i}/{len(test_CLAGN_zscores)}={i/len(test_CLAGN_zscores)*100:.3f}% of CLAGN above zscore threshold')
print(f'{k}/{len(test_CLAGN_nfd)}={k/len(test_CLAGN_nfd)*100:.3f}% of CLAGN above norm_change threshold')
print(f'{j}/{len(test_AGN_zscores)}={j/len(test_AGN_zscores)*100:.3f}% of AGN above zscore threshold')
print(f'{l}/{len(test_AGN_nfd)}={l/len(test_AGN_nfd)*100:.3f}% of AGN above norm_change threshold')

### END OF INVESTIGATION


# # A histogram of z score values & normalised flux difference values
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))  # Creates a figure with 1 row and 2 columns

# zscore_binsize = 1
# # # #CLAGN
# bins_zscores = np.arange(0, max(CLAGN_zscores)+zscore_binsize, zscore_binsize)
# ax1.hist(CLAGN_zscores, bins=bins_zscores, color='orange', edgecolor='black', label=f'binsize = {zscore_binsize}')
# ax1.axvline(median_zscore_CLAGN, linewidth=2, linestyle='--', color='black', label = f'Median = {median_zscore_CLAGN:.2f}')
# # # # #AGN
# # bins_zscores = np.arange(0, max(AGN_zscores)+zscore_binsize, zscore_binsize)
# # ax1.hist(AGN_zscores, bins=bins_zscores, color='orange', edgecolor='black', label=f'binsize = {zscore_binsize}')
# # ax1.axvline(median_zscore, linewidth=2, linestyle='--', color='black', label = f'Median = {median_zscore:.2f}')
# ax1.set_xlabel('Z Score')
# ax1.set_ylabel('Frequency')
# ax1.legend(loc='upper right')

# norm_flux_change_binsize = 0.10
# # #CLAGN
# bins_norm_flux_change = np.arange(0, max(CLAGN_norm_flux_change)+norm_flux_change_binsize, norm_flux_change_binsize)
# ax2.hist(CLAGN_norm_flux_change, bins=bins_norm_flux_change, color='blue', edgecolor='black', label=f'binsize = {norm_flux_change_binsize}')
# ax2.axvline(median_norm_flux_change_CLAGN, linewidth=2, linestyle='--', color='black', label = f'Median = {median_norm_flux_change_CLAGN:.2f}')
# # # #AGN
# # bins_norm_flux_change = np.arange(0, max(AGN_norm_flux_change)+norm_flux_change_binsize, norm_flux_change_binsize)
# # ax2.hist(AGN_norm_flux_change, bins=bins_norm_flux_change, color='blue', edgecolor='black', label=f'binsize = {norm_flux_change_binsize}')
# # ax2.axvline(median_norm_flux_change, linewidth=2, linestyle='--', color='black', label = f'Median = {median_norm_flux_change:.2f}')
# ax2.set_xlabel('Normalised Flux Difference')
# ax2.set_ylabel('Frequency')
# ax2.legend(loc='upper right')

# # #CLAGN
# plt.suptitle('Z Score & Normalised Flux Difference Distribution - Guo CLAGN', fontsize=16)
# # # #AGN
# # plt.suptitle('Z Score & Normalised Flux Difference Distribution - Parent Sample AGN', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
# plt.show()


# # # #Creating a 2d plot for normalised flux difference & z score:
# plt.figure(figsize=(12, 7))
# plt.scatter(AGN_zscores, AGN_norm_flux_change, color='blue', label='Non-CL AGN')
# plt.scatter(CLAGN_zscores, CLAGN_norm_flux_change, color='red',  label='CLAGN')
# # plt.errorbar(AGN_zscores, AGN_norm_flux_change, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_change_unc, fmt='o', color='blue', label='Non-CL AGN')
# # plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_change, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_change_unc, fmt='o', color='red',  label='CLAGN')
# plt.axhline(y=three_sigma_norm_flux_change, color='black', linestyle='--', linewidth=2, label='Threshold')
# plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 50)
# # plt.ylim(0, 5)
# # plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_norm_flux_change+AGN_norm_flux_change))
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.xlabel("Z-Score", fontsize = 26)
# plt.ylabel("Normalised Flux Difference", fontsize = 26)
# plt.title("Investigating MIR Variability in AGN", fontsize = 28)
# plt.legend(loc = 'best', fontsize=25)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of z score vs 2nd lowest flux:
# plt.figure(figsize=(12, 7))
# plt.scatter(AGN_zscores, AGN_W1_low_flux, color='blue', label='Non-CL AGN')
# plt.scatter(CLAGN_zscores, CLAGN_W1_low_flux, color='red',  label='CLAGN')
# plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2, label = 'Threshold')
# plt.xlim(0, 50)
# # plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.xlabel("Z-Score", fontsize = 26)
# plt.ylabel("W1 Band Second Lowest Flux", fontsize = 26)
# plt.title("Second Lowest W1 Flux vs Z-Score", fontsize = 28)
# plt.legend(loc = 'best', fontsize=25)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of norm flux change vs 2nd lowest flux:
# plt.figure(figsize=(12, 7))
# plt.scatter(AGN_norm_flux_change, AGN_W1_low_flux, color='blue', label='Non-CL AGN')
# plt.scatter(CLAGN_norm_flux_change, CLAGN_W1_low_flux, color='red',  label='CLAGN')
# plt.axvline(x=three_sigma_norm_flux_change, color='black', linestyle='--', linewidth=2, label = 'Threshold')
# plt.xlim(0, 1.05*max(CLAGN_norm_flux_change+AGN_norm_flux_change))
# plt.ylim(0, 1.05*max(CLAGN_W1_low_flux+AGN_W1_low_flux))
# plt.xticks(fontsize=26)
# plt.yticks(fontsize=26)
# plt.xlabel("Normalised Flux Difference", fontsize = 26)
# plt.ylabel("W1 Band Second Lowest Flux", fontsize = 26)
# plt.title("Second Lowest W1 Flux vs Normalised Flux Difference", fontsize = 28)
# plt.legend(loc = 'best', fontsize=25)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of new norm flux change vs 2nd lowest flux:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_new_norm_change, CLAGN_W1_low_flux, color='red',  label='CLAGN')
# plt.axvline(x=three_sigma_norm_flux_change, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_new_norm_change))
# plt.ylim(0, 1.05*max(CLAGN_W1_low_flux))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("New Normalised Flux Difference", fontsize = 24)
# plt.ylabel("W1 Band 2nd Lowest Flux", fontsize = 24)
# plt.title("Flux vs New Normalised Flux Difference", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of z score vs redshift:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_zscores, CLAGN_redshifts, color='red',  label='CLAGN')
# plt.scatter(CLAGN_W1_median_flux_unc, CLAGN_redshifts, color='red',  label='CLAGN')
# plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
# plt.xlim(0, 1.05*max(CLAGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_redshifts))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("Z-Score", fontsize = 24)
# plt.ylabel("Redshift", fontsize = 24)
# plt.title("Redshift vs Z-Score", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # # #Creating a 2d plot of redshift vs unc:
# plt.figure(figsize=(12, 7))
# plt.scatter(CLAGN_W1_median_flux_unc, CLAGN_redshifts, color='red',  label='CLAGN')
# plt.scatter(AGN_W1_median_flux_unc, AGN_redshifts, color='blue',  label='Non-CL AGN')
# plt.xlim(0, 1.05*max(CLAGN_W1_median_flux_unc+AGN_W1_median_flux_unc))
# plt.ylim(0, 1.05*max(CLAGN_redshifts+AGN_redshifts))
# plt.xticks(fontsize=24)
# plt.yticks(fontsize=24)
# plt.xlabel("Median Flux Uncertainty", fontsize = 24)
# plt.ylabel("Redshift", fontsize = 24)
# plt.title("Median Flux Uncertainty vs Redshift", fontsize = 24)
# plt.legend(loc = 'best', fontsize=22)
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()


# # # #Creating a 2d plot of zscore vs z score unc:
plt.figure(figsize=(12, 7))
plt.scatter(AGN_zscores, AGN_zscore_uncs, color='blue',  label='Non-CL AGN')
plt.scatter(CLAGN_zscores, CLAGN_zscore_uncs, color='red',  label='CLAGN')
plt.xlim(0, 50)
plt.ylim(0, 20)
# plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_zscore_uncs+AGN_zscore_uncs))
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("Z-Score", fontsize = 24)
plt.ylabel("Z-Score Uncertainty", fontsize = 24)
plt.title("Z-Score Uncertainty vs Z-Score", fontsize = 24)
plt.legend(loc = 'best', fontsize=22)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# # # #Creating a 2d plot of zscore vs z score unc:
plt.figure(figsize=(12, 7))
plt.scatter(AGN_norm_flux_change_unc, AGN_norm_flux_change, color='blue',  label='Non-CL AGN')
plt.scatter(CLAGN_norm_flux_change_unc, CLAGN_norm_flux_change, color='red',  label='CLAGN')
plt.xlim(0, 1.05*max(CLAGN_norm_flux_change_unc+AGN_norm_flux_change_unc))
plt.ylim(0, 1.05*max(CLAGN_norm_flux_change+AGN_norm_flux_change))
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("NFD Uncertainty", fontsize = 24)
plt.ylabel("NFD", fontsize = 24)
plt.title("NFD vs NFD Uncertainty", fontsize = 24)
plt.legend(loc = 'best', fontsize=22)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()