#%%
import xml.etree.ElementTree as ET
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
from itertools import product

#%% translate yen to euro based on historical exchange rates
# source: https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/index.en.html 
# Load and parse the XML file
xml_path = "jpy.xml"
tree = ET.parse(xml_path)
root = tree.getroot()

# XML namespaces used in the file
ns = {
    'generic': 'http://www.SDMX.org/resources/SDMXML/schemas/v2_0/message',
    'data': 'http://www.ecb.europa.eu/vocabulary/stats/exr/1'
}

# Find all observations and extract dates and exchange rates
obs_list = root.findall('.//data:Obs', ns)

# Extract date and exchange rate pairs
exchange_rates = [
    (datetime.strptime(obs.attrib['TIME_PERIOD'], "%Y-%m-%d"), float(obs.attrib['OBS_VALUE']))
    for obs in obs_list
]

# Create DataFrame from extracted exchange rates
exchange_df = pd.DataFrame(exchange_rates, columns=['date', 'yen_per_euro'])
exchange_df['euro_per_yen'] = 1/exchange_df['yen_per_euro']

exchange_df['year'] = exchange_df['date'].dt.year
exchange_df['month'] = exchange_df['date'].dt.month
monthly_avg_rates = exchange_df.groupby(['year', 'month'])['euro_per_yen'].mean().reset_index()
monthly_avg_rates
# monthly_avg_rates['euro_per_yen'].plot()
# plt.show()
#%%
# save the new dataframe
# https://www.kaggle.com/datasets/tcashion/tokyo-wholesale-tuna-prices
tuna_df = pd.read_csv("tokyo_wholesale_tuna_prices.csv")

# convert yen prices to euro based on historical exchange rate
merged_df = pd.merge(tuna_df, monthly_avg_rates, on=['year', 'month'], how='left')
merged_df['value'] = merged_df.apply(
    lambda row: row['value'] * row['euro_per_yen'] if row['measure'].lower() == 'price' else row['value'],
    axis=1
)

df = merged_df 
df.apply(pd.unique)

#%% check unique values
# df['species'].unique()
# df['state'].unique()
# df['fleet'].unique()
# df['measure'].unique()

#%% select portion of the data
# specie_type = 'Bluefin Tuna'
# specie_type = 'Southern Bluefin Tuna'
# specie_type = 'Bigeye Tuna'
# state_type = 'Fresh'
# state_type = 'Frozen'
# fleet_type = 'Japanese Fleet'
# fleet_type = 'Foreign Fleet'
# fleet_type = 'Unknown Fleet'
#%% Distribution of Prices and Demands across the dataset
df_quantity = df[df['measure'] == 'Quantity']
df_price = df[df['measure'] == 'Price']

fig, axes = plt.subplots(1,2, figsize=(14,4))
df_quantity['value'].plot(kind='hist', title='demand (tons)', ax=axes[0])
df_price['value'].plot(kind='hist', title='price (euro per kilo)', ax=axes[1])
plt.show()

#%% save the subdatasets with only the quantity and price amount 
for specie_type, state_type, fleet_type in product(['Bluefin Tuna', 'Southern Bluefin Tuna', 'Bigeye Tuna'], ['Fresh', 'Frozen'], ['Japanese Fleet', 'Foreign Fleet', 'Unknown Fleet']):
    # print(specie_type, state_type, fleet_type)

    df_quantity  = df[(df['species'] == specie_type) & 
                    (df['state'] == state_type) & 
                    (df['fleet'] == fleet_type) & 
                    (df['measure'] == 'Quantity')]

    df_price     = df[(df['species'] == specie_type) & 
                    (df['state'] == state_type) & 
                    (df['fleet'] == fleet_type) & 
                    (df['measure'] == 'Price')]
    # df_quantity
    # df_price

    if len(df_quantity) == 0 or len(df_price) == 0:
        continue

    fig, axes = plt.subplots(1,2, figsize=(14,4))
    df_quantity['value'].plot(kind='hist', title='demand (tons)', ax=axes[0])
    df_price['value'].plot(kind='hist', title='price (euro per kilo)', ax=axes[1])
    plt.show()

    # set timestamps and plot
    col_to_datetime = lambda df: pd.to_datetime(dict(year=df.year, month=df.month, day=1))
    df_quantity.set_index(col_to_datetime(df_quantity), inplace=True)
    df_price.set_index(col_to_datetime(df_price), inplace=True)
    
    df_quantity_price = pd.concat([df_quantity['value'], df_price['value']], axis=1, keys=['demand', 'price'])

    # plt.title(f'specie: {specie_type}, state: {state_type}, fleet: {fleet_type}')
    # df_quantity['value'].plot(label='quantity (metric tonnes)')
    # df_price['value'].plot(label='price (€ per kilo)')
    # plt.legend()
    # plt.show()
    
    specifier_string = f"{specie_type.split(' ')[0]}_{state_type}_{fleet_type.split(' ')[0]}Fleet"

    # df_quantity['value'].to_csv(f'tuna_demand_{specifier_string}.csv', header=['demand'], index='date')
    # df_price['value'].to_csv(f'tuna_price_{specifier_string}.csv', header=['price'], index_label='date')

    print(f"datasets/tuna_{specifier_string}.csv")
    df_quantity_price.to_csv(f'datasets/tuna_{specifier_string}.csv', index_label='date')