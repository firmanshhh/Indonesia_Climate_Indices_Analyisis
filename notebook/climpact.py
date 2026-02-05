import sys
import os
import os.path as pa
import pandas as pd
import numpy as np
import socket
from pathlib import Path
from datetime import date
import xarray as xr
import datetime
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from scipy.stats import linregress

from scipy import stats
import climpact as clim


def FIND_TXTN(df,idxtx,idxtn) :
    store=[]
    TMAX_TMIN= df[[idxtx,idxtn]].loc[df[idxtn] > df[idxtx]]
    store.append (TMAX_TMIN)
    return store

def FIND_JUMP(df, idxvar, idxdate):
    df = df.reset_index()
    store = []

    for i in range(1, len(df)):
        suhu_hari_ini = df.loc[i, idxvar]
        suhu_kemarin = df.loc[i - 1, idxvar]
        selisih = suhu_hari_ini - suhu_kemarin

        if abs(selisih) > 20:
            #kondisi.append((df.index[i], selisih))
            tanggal_kejadian = df.loc[i, idxdate]
            store.append((tanggal_kejadian,suhu_hari_ini,suhu_kemarin,selisih))
    return store

def FIND_FLATLINE(df,idxvar):
    flatline_sequences = []
    current_value = None
    current_sequence = []

    for index, row in df.iterrows():
        value = row[idxvar]
        if value == current_value:
            current_sequence.append(index)

        else:
            if len(current_sequence) >= 3:
                flatline_sequences.append(current_sequence)
            current_sequence = [index]
            current_value = value
    return flatline_sequences


def FLATLINE(df, idxvar):
    store=[]
    flatlines = FIND_FLATLINE(df, idxvar)

    for flatline_sequence in flatlines:
        # Extract the start and end indices of the flatline pattern
        start_index = flatline_sequence[0]
        end_index = flatline_sequence[-1]

        # Use Pandas indexing to select the location data within the flatline pattern
        selected_data = df.loc[start_index:end_index, [idxvar]]

        selected_data['FLATLINE'] = len(selected_data)
        result = selected_data.groupby("FLATLINE").agg("mean")
        #result['START_DATE'] = df["DATA_TIMESTAMP"].iloc[start_index]
        #result['END_DATE'] = df["DATA_TIMESTAMP"].iloc[end_index]
        result['START_DATE'] = start_index
        result['END_DATE'] = end_index
        #result = result.reset_index()

        # Append the selected data to the list
        store.append(result)

    return store

def FIND_DUPLICATE(df,subset):
    bucket=[]
    duplicates = df[df.duplicated(subset=subset, keep=False)]
    grouped_sorted = duplicates.sort_values(subset).reset_index(drop=True)
    bucket.append(grouped_sorted)
    return bucket


def DUPLICATE_COLUMN (df):
    kolom_sama = []
    num_cols = df.shape[1]
    for i in range(num_cols):
        for j in range(i + 1, num_cols):
            is_same = df.iloc[:, i].equals(df.iloc[:, j])
            if is_same:
                kolom_sama.append((df.columns[i], df.columns[j]))
    
    return kolom_sama


def DUPLICATE_ROW(df):
    baris_sama = []
    num_rows = df.shape[0]
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            is_same = df.iloc[i, 1:].equals(df.iloc[j, 1:])
            if is_same:
                baris_sama.append((df['year'][i], df['year'][j]))

    return baris_sama


def conv_matrik(df1,subset):
    matrix = []
    cats = ['January','February','March','April','May','June',
        'July','August','September','October','November','December']
    df1.reset_index(inplace=True)
    df1.rename(columns={'index': 'date'}, inplace=True)
    df1['month'] = df1['date'].dt.month
    df1['year'] = df1['date'].dt.year
    df1_transpose = df1.pivot(index='year', columns='month', values=subset).rename(columns=dict(enumerate(cats, 1)))
    #df1_transpose = df1.pivot('year','month',subset).rename(columns=dict(enumerate(cats, 1)))
    matrix.append(df1_transpose)

    return matrix

def CDD(rainfall_data, threshold):
    dry_day_count = 0
    max_dry_day_count = 0

    for rainfall in rainfall_data:
        if rainfall <= threshold:
            dry_day_count += 1
            if dry_day_count > max_dry_day_count:
                max_dry_day_count = dry_day_count
        else:
            dry_day_count = 0

    return max_dry_day_count


def CWD(rainfall_data, threshold):
    wet_day_count = 0
    max_wet_day_count = 0

    for rainfall in rainfall_data:
        if rainfall > threshold:
            wet_day_count += 1
            if wet_day_count > max_wet_day_count:
                max_wet_day_count = wet_day_count
        else:
            wet_day_count = 0

    return max_wet_day_count

def Rx1Day(rainfall_data):
    return rainfall_data.max()

def Rx5Day(rainfall_data):
    rolling_sum = rainfall_data.rolling(window=5).sum()
    return rolling_sum.max()

def Rx7Day(rainfall_data):
    rolling_sum = rainfall_data.rolling(window=7).sum()
    return rolling_sum.max()

def Rx10Day(rainfall_data):
    rolling_sum = rainfall_data.rolling(window=10).sum()
    return rolling_sum.max()


def calculate_SDII(data):
    wet_days = [value for value in data if isinstance(value, float) and value >= 1]
    if len(wet_days) > 0:
        return sum(wet_days) / len(wet_days)
    else:
        return 0
    

def calculate_climate_indices(df1):
    df1["DATA_TIMESTAMP"] = pd.to_datetime(df1["DATA_TIMESTAMP"])
    df1 = df1.set_index("DATA_TIMESTAMP")
    df1 = df1.reindex(pd.date_range(start=df1.index[0], end=df1.index[-1], freq='D'))

    df1["DTR"] = df1["TEMP_24H_TX_C"] - df1["TEMP_24H_TN_C"]

    PRECITOT = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).sum()
    HH = (df1['RAINFALL_24H_MM'] > 1).groupby(df1.index.to_period('Y')).sum()
    HH20M = (df1['RAINFALL_24H_MM'] > 20).groupby(df1.index.to_period('Y')).sum()
    HH50M = (df1['RAINFALL_24H_MM'] > 50).groupby(df1.index.to_period('Y')).sum()
    HH100M = (df1['RAINFALL_24H_MM'] > 100).groupby(df1.index.to_period('Y')).sum()
    FH20 = HH20M / HH * 100
    FH50 = HH50M / HH * 100
    FH100 = HH100M / HH * 100
    R50 = (df1['RAINFALL_24H_MM'] > 50).groupby(df1.index.to_period('Y')).sum()
    CDD = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).apply(lambda group: clim.CDD(group, threshold=0.1))
    CWD = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).apply(lambda group: clim.CWD(group, threshold=0.1))
    SDII = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).apply(lambda group: clim.calculate_SDII(group))
    RX1DAY = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).apply(lambda group: clim.Rx1Day(group))
    RX5DAY = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).apply(lambda group: clim.Rx5Day(group))
    RX7DAY = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).apply(lambda group: clim.Rx7Day(group))
    RX10DAY = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).apply(lambda group: clim.Rx10Day(group))

    Q95_CH = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).quantile(0.95)
    Q99_CH = df1['RAINFALL_24H_MM'].groupby(df1.index.to_period('Y')).quantile(0.99)

    R95P = {}
    R99P = {}

    grouped_data = df1.groupby(df1.index.to_period('Y'))

    for tahun, group_data in grouped_data:
        p95 = group_data["RAINFALL_24H_MM"].quantile(0.95)
        p99 = group_data["RAINFALL_24H_MM"].quantile(0.99)
        total_di_atas_persentil_95 = group_data[group_data["RAINFALL_24H_MM"] > p95]["RAINFALL_24H_MM"].sum()
        total_di_atas_persentil_99 = group_data[group_data["RAINFALL_24H_MM"] > p99]["RAINFALL_24H_MM"].sum()
        R95P[tahun] = total_di_atas_persentil_95
        R99P[tahun] = total_di_atas_persentil_99

    df_R95P = pd.DataFrame.from_dict(R95P, orient='index', columns=['R95P'])
    df_R99P = pd.DataFrame.from_dict(R99P, orient='index', columns=['R99P'])

    INDEK_CH = pd.DataFrame({
        'PRECTOT': round(PRECITOT,1),
        'R50': R50,
        'HH>10mm': HH,
        'HH>20mm': HH20M,
        'HH>50mm': HH50M,
        'HH>100mm': HH100M,
        'FH20': round(FH20,2),
        'FH50': round(FH50,2),
        'FH100': round(FH100,2),
        'CDD': CDD,
        'CWD': CWD,
        'SDII': round(SDII,2),
        'RX1DAY': RX1DAY,
        'RX5DAY': RX5DAY,
        'RX7DAY': RX7DAY,
        'RX10DAY': RX10DAY,
        'R95P': round(df_R95P["R95P"],1),
        'R99P': round(df_R99P["R99P"],1),
    })

    INDEK_CH['R95pTOT'] = INDEK_CH["R95P"] * 100 / INDEK_CH["PRECTOT"]
    INDEK_CH['R99pTOT'] = INDEK_CH["R99P"] * 100 / INDEK_CH["PRECTOT"]

    TXm = df1["TEMPERATURE_AVG_C"].groupby(df1.index.to_period('Y')).mean()
    TMm = df1["TEMPERATURE_AVG_C"].groupby(df1.index.to_period('Y')).max()
    TNm = df1["TEMPERATURE_AVG_C"].groupby(df1.index.to_period('Y')).min()

    TXX = df1["TEMP_24H_TX_C"].groupby(df1.index.to_period('Y')).max()
    TNX = df1["TEMP_24H_TN_C"].groupby(df1.index.to_period('Y')).max()
    TXn = df1["TEMP_24H_TX_C"].groupby(df1.index.to_period('Y')).min()
    TNn = df1["TEMP_24H_TN_C"].groupby(df1.index.to_period('Y')).min()
    DTR = df1["DTR"].groupby(df1.index.to_period('Y')).mean()
    ETR = TXX - TNn

    INDEK_T = pd.DataFrame({
        'TXM': round(TXm,1),
        'TMm': round(TMm,1),
        'TNm': round(TNm,1),

        'TXX': round(TXX,1),
        'TNX': round(TNX,1),

        'TXn': round(TXn,1),
        'TNn': round(TNn,1),
        
        'DTR': round(DTR,1),
        'ETR': round(ETR,1),
    })

    return INDEK_CH, INDEK_T

def calculate_linear_regression(df):
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=["Column", "Slope", "Intercept", "R-Value", "P-Value", "Std Error"])

    # Create an empty list to store result DataFrames
    result_dfs = []

    # Loop through the columns in your DataFrame
    for column in df.columns:
        if column == 'DATA_TIMESTAMP':
            continue  # Skip the timestamp column

        # Perform linear regression for the current column
        slope, intercept, r_value, p_value, std_err = linregress(range(len(df)), df[column])

        # Create a DataFrame with the results
        result_df = pd.DataFrame({
            "Column": [column],
            "Slope": [slope],
            "Intercept": [intercept],
            "R-Value": [r_value],
            "P-Value": [p_value],
            "Std Error": [std_err]
        })

        # Append the result DataFrame to the list
        result_dfs.append(result_df)

    # Concatenate all result DataFrames into a single DataFrame
    results_df = pd.concat(result_dfs, ignore_index=True)
    
    return results_df


def read_csv(file_path):
    df = pd.read_csv(file_path)
    df["DATA_TIMESTAMP"] = pd.to_datetime(df["DATA_TIMESTAMP"])
    df = df.set_index("DATA_TIMESTAMP")
    df = df[~df.index.duplicated(keep='first')]
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='D'), fill_value='NaN')
    df.index = df.index.map(lambda t: t.strftime('%Y-%m-%d'))

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'DATA_TIMESTAMP'}, inplace=True)

    df['RAINFALL_24H_MM'].replace(8888.0, 0, inplace=True)
    df[['TEMPERATURE_AVG_C', 'SUNSHINE_24H_H', 'REL_HUMIDITY_AVG_PC','RAINFALL_24H_MM', 'TEMP_24H_TX_C', 'TEMP_24H_TN_C']] = \
        df[['TEMPERATURE_AVG_C', 'SUNSHINE_24H_H', 'REL_HUMIDITY_AVG_PC','RAINFALL_24H_MM', 'TEMP_24H_TX_C', 'TEMP_24H_TN_C']].replace([9999, 999.9, 999], np.nan)

    df['RAINFALL_24H_MM'] = pd.to_numeric(df['RAINFALL_24H_MM'], errors='coerce')
    df['TEMPERATURE_AVG_C'] = pd.to_numeric(df['TEMPERATURE_AVG_C'], errors='coerce')
    df['TEMP_24H_TX_C'] = pd.to_numeric(df['TEMP_24H_TX_C'], errors='coerce')
    df['TEMP_24H_TN_C'] = pd.to_numeric(df['TEMP_24H_TN_C'], errors='coerce')

    df['TEMPERATURE_AVG_C'] = df['TEMPERATURE_AVG_C'].round(1)
    df['TEMP_24H_TN_C'] = df['TEMP_24H_TN_C'].round(1)
    df['TEMP_24H_TX_C'] = df['TEMP_24H_TX_C'].round(1)

    mask_avg = ~((df['TEMPERATURE_AVG_C'] > 8) & (df['TEMPERATURE_AVG_C'] < 40))
    mask_tx = ~((df['TEMP_24H_TX_C'] > 10) & (df['TEMP_24H_TX_C'] < 45))
    mask_tn = ~((df['TEMP_24H_TN_C'] > 8) & (df['TEMP_24H_TN_C'] < 35))

    df.loc[mask_avg, 'TEMPERATURE_AVG_C'] = np.nan
    df.loc[mask_tx, 'TEMP_24H_TX_C']      = np.nan
    df.loc[mask_tn, 'TEMP_24H_TN_C']      = np.nan

    return df