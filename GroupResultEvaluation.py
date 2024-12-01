# %%
import pandas as pd
from natsort import natsorted, natsort_keygen
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.core.display import HTML

# %%
blocks_df0 = pd.read_csv("./Result/OpcodeFrequency_0.csv")
# turn column 'op' as index
blocks_df0 = blocks_df0.set_index('Unnamed: 0')
blocks_df0['frequency'] = blocks_df0.filter(like='frequency').sum(axis=1)
blocks_df0['totalGas'] = blocks_df0.filter(like='totalGas').sum(axis=1)
blocks_result0 = blocks_df0[['totalGas', 'frequency']]
print(blocks_result0.shape)

blocks_df1 = pd.read_csv("./Result/OpcodeFrequency_1.csv")
# turn column 'op' as index
blocks_df1 = blocks_df1.set_index('Unnamed: 0')
blocks_df1['frequency'] = blocks_df1.filter(like='frequency').sum(axis=1)
blocks_df1['totalGas'] = blocks_df1.filter(like='totalGas').sum(axis=1)
blocks_result1 = blocks_df1[['totalGas', 'frequency']]
print(blocks_result1.shape)

blocks_df2 = pd.read_csv("./Result/OpcodeFrequency_2.csv")
# turn column 'op' as index
blocks_df2 = blocks_df2.set_index('Unnamed: 0')
blocks_df2['frequency'] = blocks_df2.filter(like='frequency').sum(axis=1)
blocks_df2['totalGas'] = blocks_df2.filter(like='totalGas').sum(axis=1)
blocks_result2 = blocks_df2[['totalGas', 'frequency']]
print(blocks_result2.shape)

blocks_df3 = pd.read_csv("./Result/OpcodeFrequency_3.csv")
# turn column 'op' as index
blocks_df3 = blocks_df3.set_index('Unnamed: 0')
blocks_df3['frequency'] = blocks_df3.filter(like='frequency').sum(axis=1)
blocks_df3['totalGas'] = blocks_df3.filter(like='totalGas').sum(axis=1)
blocks_result3 = blocks_df3[['totalGas', 'frequency']]
print(blocks_result3.shape)

# %%
blocks_result=blocks_result0+blocks_result1+blocks_result2+blocks_result3

# %%
def plot_gas_and_frequency(blocks_result, col_idx):
    # Calculate total gas usage
    sstore_gas = blocks_result.loc[['SSTORE'], 'totalGas'].sum()
    sload_gas = blocks_result.loc[['SLOAD'], 'totalGas'].sum()
    other_gas = blocks_result['totalGas'].sum() - sstore_gas - sload_gas
    gas_data = pd.Series([sstore_gas, sload_gas, other_gas], index=['SSTORE', 'SLOAD', 'Others'])

    # Calculate frequencies
    sstore_freq = blocks_result.loc[['SSTORE'], 'frequency'].sum()
    sload_freq = blocks_result.loc[['SLOAD'], 'frequency'].sum()
    other_freq = blocks_result['frequency'].sum() - sstore_freq - sload_freq
    freq_data = pd.Series([sstore_freq, sload_freq, other_freq], index=['SSTORE', 'SLOAD', 'Others'])

    # Plotting totalGas distribution
    plt.subplot(2, 4, col_idx + 1)
    plt.pie(gas_data, labels=gas_data.index, autopct='%1.2f%%', startangle=140)
    plt.title(f'Gas Usage Distribution (Dataset {col_idx + 1})')

    # Plotting frequency distribution
    plt.subplot(2, 4, col_idx + 5)
    wedges, texts = plt.pie(freq_data, startangle=140, colors=['orange', 'blue', 'gray'])
    plt.title(f'Frequency Distribution (Dataset {col_idx + 1})')

    # Calculate percentages for legend labels
    freq_percentages = (freq_data / freq_data.sum()) * 100
    legend_labels = [f'{op} - {pct:.2f}%' for op, pct in zip(freq_data.index, freq_percentages)]
    plt.legend(wedges, legend_labels, title="Operations", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Initialize the plot
fig=plt.figure(figsize=(18, 8))

# Plot for blocks_result1
plot_gas_and_frequency(blocks_result0, 0)

# Plot for blocks_result1
plot_gas_and_frequency(blocks_result1, 1)

# Plot for blocks_result2
plot_gas_and_frequency(blocks_result2, 2)

# Plot for blocks_result3
plot_gas_and_frequency(blocks_result3, 3)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('gas_and_frequency.eps',bbox_inches='tight')



# %%
# Calculate total gas usage
sstore_gas = blocks_result.loc[['SSTORE'], 'totalGas'].sum()
sload_gas = blocks_result.loc[['SLOAD'], 'totalGas'].sum()
other_gas = blocks_result['totalGas'].sum() - sstore_gas - sload_gas
gas_data = pd.Series([sstore_gas, sload_gas, other_gas], index=['SSTORE', 'SLOAD', 'Others'])

# Calculate frequencies
sstore_freq = blocks_result.loc[['SSTORE'], 'frequency'].sum()
sload_freq = blocks_result.loc[['SLOAD'], 'frequency'].sum()
other_freq = blocks_result['frequency'].sum() - sstore_freq - sload_freq
freq_data = pd.Series([sstore_freq, sload_freq, other_freq], index=['SSTORE', 'SLOAD', 'Others'])

fig=plt.figure(figsize=(9, 4))

# Plotting totalGas distribution
# plt.subplot(2, 4, 1)
plt.pie(gas_data, labels=gas_data.index, autopct='%1.2f%%', startangle=140)
# plt.title(f'Gas Usage Distribution')

# Plotting frequency distribution
# plt.subplot(2, 4, 5)
# wedges, texts = plt.pie(freq_data, startangle=140, colors=['orange', 'blue', 'gray'])
# plt.title(f'Frequency Distribution')

# Calculate percentages for legend labels
# freq_percentages = (freq_data / freq_data.sum()) * 100
# legend_labels = [f'{op} - {pct:.2f}%' for op, pct in zip(freq_data.index, freq_percentages)]
# plt.legend(wedges, legend_labels, title="Operations", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('gas_distribution.eps',bbox_inches='tight')


# %%
# delete row CALL/CREATE of blocks_result
blocks_result = blocks_result.drop(['CALL/CREATE'])


# %%
blocks_result

# %%
# write code to remove NaN value in blocks_result
blocks_result = blocks_result.dropna()
# order blocks_result by totalGas
blocks_result = blocks_result.sort_values(by='totalGas', ascending=True)

fig=plt.figure(figsize=(9, 4))

# Plotting totalGas distribution
# plt.subplot(2, 4, 1)

blocks_result['percentage'] = blocks_result['totalGas'] / blocks_result['totalGas'].sum() * 100

labels = [
    label if percentage >= 4 else ''
    for label, percentage in zip(blocks_result.index, blocks_result['percentage'])
]
wedges, texts, autotexts = plt.pie(blocks_result['totalGas'], labels=labels, autopct='%1.2f%%', startangle=140)

for autotext, percentage in zip(autotexts, blocks_result['percentage']):
    if percentage < 4:
        autotext.set_text('')

# legend_labels = [f'{op} - {pct:.2f}' for op, pct in zip(blocks_result.index, blocks_result['totalGas'])]
# plt.legend(wedges, legend_labels, title="Operations", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# plt.title(f'Gas Usage Distribution')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('gas_distribution.eps',bbox_inches='tight')


# %%
# write code to remove NaN value in blocks_result
blocks_result = blocks_result.dropna()
# order blocks_result by totalGas
blocks_result = blocks_result.sort_values(by='frequency', ascending=True)

fig=plt.figure(figsize=(9, 4))

# Plotting totalGas distribution
# plt.subplot(2, 4, 1)

blocks_result['percentage'] = blocks_result['frequency'] / blocks_result['frequency'].sum() * 100

labels = [
    label if percentage >= 4 else (f'{label}\n({percentage:.2f}%)' if (label=='SSTORE' or label=='SLOAD') else '')
    for label, percentage in zip(blocks_result.index, blocks_result['percentage'])
]
wedges, texts, autotexts = plt.pie(blocks_result['frequency'], labels=labels, autopct='%1.2f%%', startangle=140)

for autotext, percentage in zip(autotexts, blocks_result['percentage']):
    if percentage < 4:
        autotext.set_text('')

# legend_labels = [f'{op} - {pct:.2f}' for op, pct in zip(blocks_result.index, blocks_result['totalGas'])]
# plt.legend(wedges, legend_labels, title="Operations", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# plt.title(f'Number of occurrences of each opcode')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('occurrences.eps',bbox_inches='tight')


# %%
# Calculate total gas usage
sstore_gas = blocks_result.loc[['SSTORE'], 'totalGas'].sum()
sload_gas = blocks_result.loc[['SLOAD'], 'totalGas'].sum()
other_gas = blocks_result['totalGas'].sum() - sstore_gas - sload_gas
gas_data = pd.Series([sstore_gas, sload_gas, other_gas], index=['SSTORE', 'SLOAD', 'Others'])

# Calculate frequencies
sstore_freq = blocks_result.loc[['SSTORE'], 'frequency'].sum()
sload_freq = blocks_result.loc[['SLOAD'], 'frequency'].sum()
other_freq = blocks_result['frequency'].sum() - sstore_freq - sload_freq
freq_data = pd.Series([sstore_freq, sload_freq, other_freq], index=['SSTORE', 'SLOAD', 'Others'])

fig=plt.figure(figsize=(9, 4))

# Plotting totalGas distribution
# plt.subplot(2, 4, 1)
plt.pie(gas_data, labels=gas_data.index, autopct='%1.2f%%', startangle=140)
# plt.title(f'Gas Usage Distribution')

# Plotting frequency distribution
# plt.subplot(2, 4, 5)
# wedges, texts = plt.pie(freq_data, startangle=140, colors=['orange', 'blue', 'gray'])
# plt.title(f'Frequency Distribution')

# Calculate percentages for legend labels
# freq_percentages = (freq_data / freq_data.sum()) * 100
# legend_labels = [f'{op} - {pct:.2f}%' for op, pct in zip(freq_data.index, freq_percentages)]
# plt.legend(wedges, legend_labels, title="Operations", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('gas_distribution.eps',bbox_inches='tight')


# %%
# Calculate total gas usage
sstore_gas = blocks_result.loc[['SSTORE'], 'totalGas'].sum()
sload_gas = blocks_result.loc[['SLOAD'], 'totalGas'].sum()
other_gas = blocks_result['totalGas'].sum() - sstore_gas - sload_gas
gas_data = pd.Series([sstore_gas, sload_gas, other_gas], index=['SSTORE', 'SLOAD', 'Others'])

# Calculate frequencies
sstore_freq = blocks_result.loc[['SSTORE'], 'frequency'].sum()
sload_freq = blocks_result.loc[['SLOAD'], 'frequency'].sum()
other_freq = blocks_result['frequency'].sum() - sstore_freq - sload_freq
freq_data = pd.Series([sstore_freq, sload_freq, other_freq], index=['SSTORE', 'SLOAD', 'Others'])

fig=plt.figure(figsize=(9, 4))

# Plotting totalGas distribution
# plt.subplot(2, 4, 1)
# plt.pie(gas_data, labels=gas_data.index, autopct='%1.2f%%', startangle=140)
# plt.title(f'Gas Usage Distribution')

# Plotting frequency distribution
# plt.subplot(2, 4, 5)
wedges, texts = plt.pie(freq_data, startangle=140, colors=['orange', 'blue', 'gray'])
# plt.title(f'Frequency Distribution')

# Calculate percentages for legend labels
freq_percentages = (freq_data / freq_data.sum()) * 100
legend_labels = [f'{op} - {pct:.2f}%' for op, pct in zip(freq_data.index, freq_percentages)]
plt.legend(wedges, legend_labels, title="Operations", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('frequency.eps',bbox_inches='tight')


# %%
fig = plt.figure()
fig.set_figheight(3)
fig.set_figwidth(3)

blocks_result['totalGas'].plot(kind='line',color='black', marker = 'd')

# %% [markdown]
# ### Save Gas

# %%
df0 = pd.read_csv('./Result/CacheResult_00.csv')
df1 = pd.read_csv('./Result/CacheResult_11.csv')
df2 = pd.read_csv('./Result/CacheResult_22.csv')
df3 = pd.read_csv('./Result/CacheResult_33.csv')

# %%
df0

# %%
def load_data(df):
    def hex_storageKey(hex_str):
        hex_str = str(hex_str)
        
        if not (hex_str.startswith('0x')):
            hex_str = int(hex_str)
            hex_str = str(hex(hex_str))
        # Remove leading zeros and ensure at least one character remains
        # Ensure hex_str is a string
        hex_str = hex_str[2:]
        shortened = hex_str.lstrip('0')
        return '0x' + (shortened if shortened else '0')

    df['storageKey'] = df['storageKey'].apply(hex_storageKey)

    # Add new column isCold = 1 if in that row, gasCost > 2000, otherwise it is 0
    df['isCold'] = np.where(df['gasCost'] > 2000, 1, 0)
    df['isCacheHit'] = np.where((df['isCold'] == 1) & (df['newGasCost'] < 2000), 1, 0)

    df = df.reset_index(drop=True)
    return df

df0 = load_data(df0)
df1 = load_data(df1)
df2 = load_data(df2)
df3 = load_data(df3)

# %%
def check_num_operation(df, blocks_df):
    print("Number of SSTORE", len(df[df['op'] == 'SSTORE']))
    print("Number of SLOAD", len(df[df['op'] == 'SLOAD']))

    print("Number of SSTORE 1: ", blocks_df.loc['SSTORE']['frequency'])
    print("Number of SLOAD 1: ", blocks_df.loc['SLOAD']['frequency'])

check_num_operation(df0, blocks_result0)
check_num_operation(df1, blocks_result1)
check_num_operation(df2, blocks_result2)
check_num_operation(df3, blocks_result3)

# %%
def df_info(df):
    print("Transaction: ", df['transaction'].nunique())
    print("Contract: ", df['contract'].nunique())
    print("SSTORE cold: ", len(df[(df['isCold'] == 1) & (df['op']  == 'SSTORE')]))
    print("SSTORE warm: ", len(df[(df['isCold'] == 0) & (df['op']  == 'SSTORE')]))
    print("SLOAD cold: ", len(df[(df['isCold'] == 1 ) & (df['op'] == 'SLOAD')]))
    print("SLOAD warm: ", len(df[(df['isCold'] == 0 ) & (df['op'] == 'SLOAD')]))

df_info(df3)


# %%
# concat df0, df1, df2, df3 to df
df = pd.concat([df0, df1, df2, df3], ignore_index=True)

# %%
df_info(df)

# %% [markdown]
# ### Calculate Hit/Miss rate

# %%
def draw_hit_rate(ax, df, title_suffix):
    hit_rate_sload = df[df['op'] == 'SLOAD']['isCacheHit'].sum() / df[df['op'] == 'SLOAD']['isCold'].sum()
    hit_rate_sstore = df[df['op'] == 'SSTORE']['isCacheHit'].sum() / df[df['op'] == 'SSTORE']['isCold'].sum()

    # Data for plotting
    hit_rates = [hit_rate_sload, hit_rate_sstore]
    labels = ['SLOAD', 'SSTORE']

    # Create bar chart
    bars = ax.bar(labels, hit_rates, color=['blue', 'green'])

    # Adding titles and labels
    ax.set_title(f'Hit Rates for SLOAD and SSTORE ({title_suffix})')
    ax.set_xlabel('Operation')
    ax.set_ylabel('Hit Rate')

    # Adding the hit rate values in the middle of the bars
    for bar, rate in zip(bars, hit_rates):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval / 2, f"{rate:.2%}", ha='center', va='center', color='white', fontweight='bold')

# Create a figure with 3 subplots
fig, axs = plt.subplots(1, 4, figsize=(18, 6))

# Draw hit rates for each dataset in a separate subplot
draw_hit_rate(axs[0], df0, 'Dataset 0')
draw_hit_rate(axs[1], df1, 'Dataset 1')
draw_hit_rate(axs[2], df2, 'Dataset 2')
draw_hit_rate(axs[3], df3, 'Dataset 3')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# %%
hit_rate_sload = df[df['op'] == 'SLOAD']['isCacheHit'].sum() / df[df['op'] == 'SLOAD']['isCold'].sum()
hit_rate_sstore = df[df['op'] == 'SSTORE']['isCacheHit'].sum() / df[df['op'] == 'SSTORE']['isCold'].sum()

cold_sload = df[df['op'] == 'SLOAD']['isCold'].sum()
cold_sstore = df[df['op'] == 'SSTORE']['isCold'].sum()

cache_hit_sload = df[df['op'] == 'SLOAD']['isCacheHit'].sum()
cache_hit_sstore = df[df['op'] == 'SSTORE']['isCacheHit'].sum()

# Data for plotting
cold_num = [cold_sstore, cold_sload]
cache_hit_num = [cache_hit_sstore, cache_hit_sload]

hit_rates = [hit_rate_sstore, hit_rate_sload]
labels = ['SSTORE', 'SLOAD']




# %%
# Create bar chart for hit_rate_sload and hit_rate_sstore
fig=plt.figure(figsize=(4, 3))

# Create bar chart
# bars = plt.bar(labels, hit_rates, color=['darkorange', 'steelblue'], width=0.5)

bar_width = 0.5
ind=[2,3.5]
ind2=[2.5,4]

# Create bar chart
p2 = plt.bar(ind, cold_num, bar_width, label='Cold access', color='lightgray')
p1 = plt.bar(ind2, cache_hit_num, bar_width, label='Hit', color='steelblue')


# plt.title(f'Hit Rates for SLOAD and SSTORE')
plt.xlabel('Operation')
plt.ylabel('Count')

# Adding the hit rate values in the middle of the bars
# for bar, rate in zip(bars, hit_rates):
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, yval / 2, f"{rate:.2%}", ha='center', va='center', color='white', fontweight='bold')


plt.xticks([2.25,3.75], ('SSTORE', 'SLOAD'))

# Adding the gas cost values on top of the bars
# for i, (total, saved, new) in enumerate(zip(total_gas_costs, saved_gas_costs, new_gas_costs)):
#     plt.text(ind2[i], new, f'{saved:,}', ha='center', va='center', color='white', fontweight='bold')

# Adding percentage labels for saved gas cost
plt.text(ind2[0]+0.05, cache_hit_num[0]+2e5, f'Hit Rate:\n{hit_rate_sstore*100:.2f}%\n', ha='center', va='center', fontsize=8)
plt.text(ind2[1]+0.05, cache_hit_num[1]+2e5, f'Hit Rate:\n{hit_rate_sload*100:.2f}%\n', ha='center', va='center', fontsize=8)

# Add legend
plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('hit-rate.eps',bbox_inches='tight')

# %%
# Create bar chart for hit_rate_sload and hit_rate_sstore
fig=plt.figure(figsize=(4, 3))

# Create bar chart
# bars = plt.bar(labels, hit_rates, color=['darkorange', 'steelblue'], width=0.5)

bar_width = 0.5
ind=[2,3.5]
ind2=[2.5,4]

# Create bar chart
p1 = plt.bar(ind, [cold_sstore, cache_hit_sstore], bar_width, label='SSTORE', color='darkorange')
p2 = plt.bar(ind2, [cold_sload, cache_hit_sload], bar_width, label='SLOAD', color='steelblue')


# plt.title(f'Hit Rates for SLOAD and SSTORE')
plt.ylabel('Number of operations')

# Adding the hit rate values in the middle of the bars
# for bar, rate in zip(bars, hit_rates):
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, yval / 2, f"{rate:.2%}", ha='center', va='center', color='white', fontweight='bold')


plt.xticks([2.25,3.75], ('Cold access', 'Hit'))

# Adding the gas cost values on top of the bars
# for i, (total, saved, new) in enumerate(zip(total_gas_costs, saved_gas_costs, new_gas_costs)):
#     plt.text(ind2[i], new, f'{saved:,}', ha='center', va='center', color='white', fontweight='bold')

# Adding percentage labels for saved gas cost
plt.text(ind[1], cache_hit_sstore+2e5, f'\n{hit_rate_sstore*100:.2f}%\n', ha='center', va='center', fontsize=8)
plt.text(ind2[1], cache_hit_sload+2e5, f'\n{hit_rate_sload*100:.2f}%\n', ha='center', va='center', fontsize=8)

# Add legend
plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('hit-rate.eps',bbox_inches='tight')

# %%



# %%
df0

# %% [markdown]
# ### Gas saved result

# %% [markdown]
# ### Saved for each operation

# %%
def plot_stacked_gas_cost_comparison(ax, df, title_suffix):
    # Sum data for SSTORE operation
    sstore_gas_cost = df[df['op'] == 'SSTORE']['gasCost'].sum()
    sstore_new_gas_cost = df[df['op'] == 'SSTORE']['newGasCost'].sum()
    sstore_saved_gas_cost = sstore_gas_cost - sstore_new_gas_cost
    sstore_percentage = (sstore_saved_gas_cost / sstore_gas_cost) * 100

    # Sum data for SLOAD operation
    sload_gas_cost = df[df['op'] == 'SLOAD']['gasCost'].sum()
    sload_new_gas_cost = df[df['op'] == 'SLOAD']['newGasCost'].sum()
    sload_saved_gas_cost = sload_gas_cost - sload_new_gas_cost
    sload_percentage = (sload_saved_gas_cost / sload_gas_cost) * 100

    # Data for plotting
    categories = ['SSTORE', 'SLOAD']
    total_gas_costs = [sstore_gas_cost, sload_gas_cost]
    saved_gas_costs = [sstore_saved_gas_cost, sload_saved_gas_cost]
    new_gas_costs = [sstore_new_gas_cost, sload_new_gas_cost]

    # Create stacked bar chart
    bar_width = 0.5
    p1 = ax.bar(categories, new_gas_costs, bar_width, label='New Gas Cost', color='blue')
    p2 = ax.bar(categories, saved_gas_costs, bar_width, bottom=new_gas_costs, label='Saved Gas Cost', color='green')

    # Adding titles and labels
    ax.set_title(f'Gas Cost Comparison ({title_suffix})')
    ax.set_ylabel('Gas Cost')

    # Adding the gas cost values on top of the bars
    for i, (total, saved, new) in enumerate(zip(total_gas_costs, saved_gas_costs, new_gas_costs)):
        ax.text(i, new / 2, f'{new:,}', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, new + saved / 2, f'{saved:,}', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, total, f'{total:,}', ha='center', va='bottom')

    # Adding percentage labels for saved gas cost
    ax.text(0, sstore_gas_cost / 2, f'Saved: {sstore_percentage:.2f}%', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1, sload_gas_cost / 2, f'Saved: {sload_percentage:.2f}%', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Add legend
    ax.legend()

# Initialize the plot
fig, axs = plt.subplots(1, 4, figsize=(18, 6))

# Plot for each dataset
plot_stacked_gas_cost_comparison(axs[0], df0, 'Dataset 0')
plot_stacked_gas_cost_comparison(axs[1], df1, 'Dataset 1')
plot_stacked_gas_cost_comparison(axs[2], df2, 'Dataset 2')
plot_stacked_gas_cost_comparison(axs[3], df3, 'Dataset 3')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# %%


# %% [markdown]
# ### Overall Saved Gas

# %%
def plot_overall_comparison(ax, df, blocks_result, title_suffix):
    # Sum data for all operations
    gasCost = df['gasCost'].sum()
    newGasCost = df['newGasCost'].sum()
    savedGasCost = gasCost - newGasCost
    percentage = (savedGasCost / gasCost)*100

    blockGasCost = blocks_result['totalGas'].sum()
    blockNewGasCost = blockGasCost - savedGasCost
    blockPercentage = (savedGasCost / blockGasCost)*100


    # Data for plotting
    categories = ['Storage Gas', 'Total Gas']
    total_values = [gasCost, blockGasCost]
    saved_values = [savedGasCost, savedGasCost]
    new_values = [newGasCost, blockNewGasCost]

    # Create stacked bar chart
    bar_width = 0.5
    p1 = ax.bar(categories, new_values, bar_width, label='New Gas Cost', color='blue')
    p2 = ax.bar(categories, saved_values, bar_width, bottom=new_values, label='Saved Gas Cost', color='green')

    # Adding titles and labels
    ax.set_title(f'Gas Cost Comparison ({title_suffix})')
    ax.set_ylabel('Gas Cost')

    # Adding the gas cost values on top of the bars
    for i, (total, saved, new) in enumerate(zip(total_values, saved_values, new_values)):
        ax.text(i, new / 2, f'{new:,}', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, new + saved / 2, f'{saved:,}', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, total, f'{total:,}', ha='center', va='bottom')

    # Adding percentage labels for saved gas cost
    ax.text(0, gasCost / 2, f'Saved: {percentage:.2f}%', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
    ax.text(1, blockGasCost / 2, f'Saved: {blockPercentage:.2f}%', ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Add legend
    ax.legend()

# Initialize the plot
fig, axs = plt.subplots(1, 4, figsize=(18, 6))

# Plot for each dataset
plot_overall_comparison(axs[0], df0, blocks_result0, 'Dataset 0')
plot_overall_comparison(axs[1], df1, blocks_result1, 'Dataset 1')
plot_overall_comparison(axs[2], df2, blocks_result2, 'Dataset 2')
plot_overall_comparison(axs[3], df3, blocks_result3, 'Dataset 3')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# %%
sstore_gas_cost = df[df['op'] == 'SSTORE']['gasCost'].sum()
sstore_new_gas_cost = df[df['op'] == 'SSTORE']['newGasCost'].sum()
sstore_saved_gas_cost = sstore_gas_cost - sstore_new_gas_cost
sstore_percentage = (sstore_saved_gas_cost / sstore_gas_cost) * 100

# Sum data for SLOAD operation
sload_gas_cost = df[df['op'] == 'SLOAD']['gasCost'].sum()
sload_new_gas_cost = df[df['op'] == 'SLOAD']['newGasCost'].sum()
sload_saved_gas_cost = sload_gas_cost - sload_new_gas_cost
sload_percentage = (sload_saved_gas_cost / sload_gas_cost) * 100

# Data for plotting
total_gas_costs = [sstore_gas_cost, sload_gas_cost]
saved_gas_costs = [sstore_saved_gas_cost, sload_saved_gas_cost]
new_gas_costs = [sstore_new_gas_cost, sload_new_gas_cost]

gasCost = df['gasCost'].sum()
newGasCost = df['newGasCost'].sum()
savedGasCost = gasCost - newGasCost
percentage = (savedGasCost / gasCost)*100

blockGasCost = blocks_result['totalGas'].sum()
blockNewGasCost = blockGasCost - savedGasCost
blockPercentage = (savedGasCost / blockGasCost)*100


# Data for plotting
categories = ['SSTORE', 'SLOAD', 'Storage Gas', 'Total Gas']
total_values = [sstore_gas_cost, sload_gas_cost, gasCost, blockGasCost]
saved_values = [sstore_saved_gas_cost, sload_saved_gas_cost, savedGasCost, savedGasCost]
new_values = [sstore_new_gas_cost, sload_new_gas_cost, newGasCost, blockNewGasCost]

# %%
fig=plt.figure(figsize=(5, 3.5))

# Create stacked bar chart
bar_width = 0.5
ind=[2,3.5, 5, 6.5]
ind2=[2.5,4, 5.5, 7]

# Create bar chart
p2 = plt.bar(ind, total_values, bar_width, label='Original Gas Cost', color='lightgray')
p1 = plt.bar(ind2, new_values, bar_width, label='New Gas Cost', color='steelblue')

# Adding titles and labels
# plt.title(f'Gas Cost Comparison')
plt.ylabel('Gas Cost')
plt.xticks([2.25,3.75, 5.25, 6.75], ('SSTORE', 'SLOAD', 'Storage Gas', 'Total Gas'))

# Adding the gas cost values on top of the bars
# for i, (total, saved, new) in enumerate(zip(total_gas_costs, saved_gas_costs, new_gas_costs)):
#     plt.text(ind2[i], new, f'{saved:,}', ha='center', va='center', color='white', fontweight='bold')

# Adding percentage labels for saved gas cost
plt.text(ind2[0]+0.05, sstore_new_gas_cost+3e9, f'\n{sstore_percentage:.2f}%\n saved', ha='center', va='center', fontsize=8)
plt.text(ind2[1]+0.05, sload_new_gas_cost+3e9, f'\n{sload_percentage:.2f}%\n saved', ha='center', va='center', fontsize=8)
plt.text(ind2[2]+0.05, newGasCost+3e9, f'{percentage:.2f}%\n saved', ha='center', va='center', fontsize=8)
plt.text(ind2[3]+0.05, blockNewGasCost+3e9, f'\n{blockPercentage:.2f}%\n saved', ha='center', va='center', fontsize=8)

# Add legend
plt.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
fig.savefig('gas-saving.eps',bbox_inches='tight')




# %%
total_gas_costs
# new_gas_costs

# %% [markdown]
# ## Block result analysis

# %%
def get_block_df(df):
    block_df = df.groupby('block').agg(
        numOfStorageKey=('storageKey', 'nunique'),
        noSLOAD=('op', lambda x: (x == 'SLOAD').sum()),
        noSSTORE=('op', lambda x: (x == 'SSTORE').sum()),
        originalGas=('gasCost', 'sum'),
        newGasCost=('newGasCost', 'sum'),
        noOfTransaction=('transaction', 'nunique'),
        noOfContract=('contract', 'nunique')
    )

    # Calculate savedGas and savedPercentage
    block_df['savedGas'] = block_df['originalGas'] - block_df['newGasCost']
    block_df['savedPercentage'] = (block_df['savedGas'] / block_df['originalGas']) * 100

    # Reset index so contract be a column rather than an index
    block_df = block_df.reset_index()
    return block_df

block_df1 = get_block_df(df1)
block_df2 = get_block_df(df2)
block_df3 = get_block_df(df3)

# %%
block_df1

# %%
def plot_gas_saving(block_df):
    # Histogram for 4 column originalGas and savedPercentage
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # Stack bar plot for newGasCost and savedGas
    block_df[['newGasCost', 'savedGas']].plot(kind='bar', stacked=True, ax=axes[0], color=['steelblue', 'blue'])

    # Set the x-axis to range and hide labels
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])

    axes[0].set_title('Comparison of Original Gas and Saved Gas')
    axes[0].set_ylabel('Gas Cost')
    axes[0].set_xlabel('Block')
    axes[0].legend(['Gas after apply cache', 'Saved Gas (Theroetical)'])

    # Plot histogram for savedPercentage
    sns.histplot(data=block_df, x='savedPercentage', kde=True, ax=axes[1], bins=250)
    axes[1].set_title('Distribution of Saved Percentage for Blocks')
    axes[1].set_xlabel('Saved Percentage (%)')
    axes[1].set_ylabel('Number of Blocks')

    # Adjust layout
    plt.tight_layout()

    # Display plot
    plt.show()

plot_gas_saving(block_df1)
plot_gas_saving(block_df2)
plot_gas_saving(block_df3)

# %%
def plot_gas_saving1(block_df):
    # Histogram for 4 column originalGas and savedPercentage
    fig, axes = plt.subplots(1, 1, figsize=(9, 4))

    # Plot histogram for savedPercentage
    sns.histplot(data=block_df, x='savedPercentage', kde=True, ax=axes, bins=250)
    axes.set_title('Distribution of Saved Percentage for Blocks')
    axes.set_xlabel('Saved Percentage (%)')
    axes.set_ylabel('Number of Blocks')

    # Adjust layout
    plt.tight_layout()

    # Display plot
    plt.show()
combined_df = pd.concat([df0, df1, df2, df3])
plot_gas_saving1(get_block_df(combined_df))

# %%
combined_df = pd.concat([df0, df1, df2, df3])
combined_df = get_block_df(combined_df)

# %%

# Histogram for 4 column originalGas and savedPercentage
fig, axes = plt.subplots(1, 1, figsize=(6, 4))

# Plot histogram for savedPercentage
sns.histplot(data=combined_df, x='savedPercentage', kde=True, ax=axes, bins=125, color='skyblue')
# axes.set_title('Distribution of Saved Percentage for Blocks')
axes.set_xlabel('Gas-Saved Percentage (%)')
axes.set_ylabel('Number of Blocks')
axes.set_xlim(7, 42)

# Adjust layout
plt.tight_layout()

# Display plot
plt.show()
fig.savefig('gas-saving-histogram.eps',bbox_inches='tight')




# %%
def show_hit_rate_distribution(data):
    # Plot distributions
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(data['sload_hit_rate'], kde=True, bins=30)
    plt.title('Distribution of hit rate for SLOAD')
    plt.xlabel('Hit rate')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(data['sstore_hit_rate'], kde=True, bins=30)
    plt.title('Distribution of hit rate for SSTORE')
    plt.xlabel('Hit rate')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### Contract analysis

# %%
def get_contract_df(df):
    # Precompute the conditions
    df['ColdSSTORE'] = (df['isCold'] == 1) & (df['op'] == 'SSTORE')
    df['ColdSLOAD'] = (df['isCold'] == 1) & (df['op'] == 'SLOAD')
    df['WarmSSTORE'] = (df['isCold'] == 0) & (df['op'] == 'SSTORE')
    df['WarmSLOAD'] = (df['isCold'] == 0) & (df['op'] == 'SLOAD')
    df['CacheHitSSTORE'] = (df['isCacheHit'] == 1) & (df['op'] == 'SSTORE')
    df['CacheHitSLOAD'] = (df['isCacheHit'] == 1) & (df['op'] == 'SLOAD')
    
    contract_df = df.groupby('contract').agg(
        numOfStorageKey=('storageKey', 'nunique'),
        noSLOAD=('op', lambda x: (x == 'SLOAD').sum()),
        noSSTORE=('op', lambda x: (x == 'SSTORE').sum()),
        originalGas=('gasCost', 'sum'),
        newGasCost=('newGasCost', 'sum'),
        noOfTransaction=('transaction', 'nunique'),
        noOfBlock=('block', 'nunique'),
        ColdSSTORE=('ColdSSTORE', 'sum'),
        ColdSLOAD=('ColdSLOAD', 'sum'),
        WarmSSTORE=('WarmSSTORE', 'sum'),
        WarmSLOAD=('WarmSLOAD', 'sum'),
        CacheHitSSTORE=('CacheHitSSTORE', 'sum'),
        CacheHitSLOAD=('CacheHitSLOAD', 'sum'),
    )


    # Calculate savedGas and savedPercentage
    contract_df['savedGas'] = contract_df['originalGas'] - contract_df['newGasCost']
    contract_df['savedPercentage'] = (contract_df['savedGas'] / contract_df['originalGas']) * 100
    contract_df['HitRateSSTORE'] = (contract_df['CacheHitSSTORE'] / contract_df['ColdSSTORE']) * 100
    contract_df['HitRateSLOAD'] = (contract_df['CacheHitSLOAD'] / contract_df['ColdSLOAD']) * 100

    contract_df['HitRateSSTORE'] = contract_df['HitRateSSTORE'].fillna(0)
    contract_df['HitRateSLOAD'] = contract_df['HitRateSLOAD'].fillna(0)

    # Reset index so contract be a column rather than an index
    contract_df = contract_df.reset_index()

    return contract_df

combined_df = pd.concat([df0, df1, df2, df3])
contract_df = get_contract_df(combined_df)
contract_df

# %%
contract_df.describe()

# %%
percent = contract_df.sort_values(by=['savedPercentage'], ascending=False).iloc[:10]
percent[['contract', 'HitRateSSTORE', 'ColdSSTORE', 'WarmSSTORE', 'HitRateSLOAD', 'ColdSLOAD', 'WarmSLOAD', 'originalGas', 'savedGas', 'savedPercentage']]

# %%
percent = contract_df.sort_values(by=['savedGas'], ascending=False).iloc[:10]
percent[['contract', 'HitRateSSTORE', 'ColdSSTORE', 'WarmSSTORE', 'HitRateSLOAD', 'ColdSLOAD', 'WarmSLOAD', 'originalGas', 'savedGas', 'savedPercentage']]

# %%
contract1 = combined_df[combined_df['contract'] == '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48']
contract2 = combined_df[combined_df['contract'] == '0x5b2e4a700dfbc560061e957edec8f6eeeb74a320']
contract3 = combined_df[combined_df['contract'] == '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2']

contract1[['storageKey', 'op', 'transaction', 'gasCost', 'storageValue', 'newGasCost']]

# %%
contract3_1 = contract3[contract3['CacheHitSSTORE'] == 1]
contract3[['storageKey', 'op', 'transaction', 'gasCost', 'storageValue', 'newGasCost', 'block']].groupby('storageKey').apply(lambda x: x.sort_values('transaction')).iloc[:20]


