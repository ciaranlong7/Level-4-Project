import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Quantifying change data
CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_2nd_biggest_smallest.csv')
# CLAGN_quantifying_change_data = pd.read_csv('CLAGN_Quantifying_Change_just_MIR_2nd_biggest_smallest_mean_uncs.csv')
print(f'Number of CLAGN: {len(CLAGN_quantifying_change_data)}')
CLAGN_zscores = CLAGN_quantifying_change_data.iloc[:, 17].tolist()  # 18th column
CLAGN_zscore_uncs = CLAGN_quantifying_change_data.iloc[:, 18].tolist()
CLAGN_norm_flux_change = CLAGN_quantifying_change_data.iloc[:, 19].tolist()
CLAGN_norm_flux_change_unc = CLAGN_quantifying_change_data.iloc[:, 20].tolist()

AGN_quantifying_change_data = pd.read_csv('AGN_Quantifying_Change_just_MIR_2nd_biggest_smallest.csv')
# AGN_quantifying_change_data = pd.read_csv('AGN_Quantifying_Change_just_MIR_2nd_biggest_smallest_mean_uncs.csv')
print(f'Number of AGN: {len(AGN_quantifying_change_data)}')
AGN_zscores = AGN_quantifying_change_data.iloc[:, 17].tolist()
AGN_zscore_uncs = AGN_quantifying_change_data.iloc[:, 18].tolist()
AGN_norm_flux_change = AGN_quantifying_change_data.iloc[:, 19].tolist()
AGN_norm_flux_change_unc = AGN_quantifying_change_data.iloc[:, 20].tolist()

#want the median value of the random sample of AGN
median_norm_flux_change = np.nanmedian(AGN_norm_flux_change)
median_norm_flux_change_unc = np.nanmedian(AGN_norm_flux_change_unc)
three_sigma_norm_flux_change = median_norm_flux_change + 3*median_norm_flux_change_unc
median_zscore = np.nanmedian(AGN_zscores)
median_zscore_unc = np.nanmedian(AGN_zscore_uncs)
three_sigma_zscore = median_zscore + 3*median_zscore_unc
print(f'3\u03C3 significance for norm flux change = {three_sigma_norm_flux_change}')
print(f'3\u03C3 significance for z score = {three_sigma_zscore}')

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

print(f'{i/len(CLAGN_zscores)*100:.2f}% of CLAGN above zscore threshold')
print(f'{k/len(CLAGN_norm_flux_change)*100:.2f}% of CLAGN above norm_change threshold')
print(f'{j/len(AGN_zscores)*100:.2f}% of AGN above zscore threshold')
print(f'{l/len(AGN_norm_flux_change)*100:.2f}% of AGN above norm_change threshold')

# # A histogram of z score values & normalised flux change values
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
# ax2.set_xlabel('Normalised Flux Change')
# ax2.set_ylabel('Frequency')
# ax2.legend(loc='upper right')

# # #CLAGN
# plt.suptitle('Z Score & Normalised Flux Change Distribution - Guo CLAGN', fontsize=16)
# # # #AGN
# # plt.suptitle('Z Score & Normalised Flux Change Distribution - Parent Sample AGN', fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the main title
# plt.show()


# # # #Creating a 2d plot for normalised flux change & z score:
# plt.figure(figsize=(7, 7)) #square figure
# plt.scatter(AGN_zscores, AGN_norm_flux_change, color='blue', label='Parent Sample AGN')
# plt.scatter(CLAGN_zscores, CLAGN_norm_flux_change, color='red',  label='Guo CLAGN')
# # plt.errorbar(AGN_zscores, AGN_norm_flux_change, xerr=AGN_zscore_uncs, yerr=AGN_norm_flux_change_unc, fmt='o', color='blue', label='Parent Sample AGN')
# # plt.errorbar(CLAGN_zscores, CLAGN_norm_flux_change, xerr=CLAGN_zscore_uncs, yerr=CLAGN_norm_flux_change_unc, fmt='o', color='red',  label='Guo CLAGN')
# plt.axhline(y=three_sigma_norm_flux_change, color='black', linestyle='--', linewidth=2, label=u'3\u03C3 significance')
# plt.axvline(x=three_sigma_zscore, color='black', linestyle='--', linewidth=2)
# # plt.xlim(0, 40)
# # plt.ylim(0, 5)
# plt.xlim(0, 1.05*max(CLAGN_zscores+AGN_zscores))
# plt.ylim(0, 1.05*max(CLAGN_norm_flux_change+AGN_norm_flux_change))
# plt.xlabel("Z Score")
# plt.ylabel("Normalised Flux Change")
# plt.title("Quantifying MIR Variability in AGN")
# plt.legend(loc = 'best')
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()