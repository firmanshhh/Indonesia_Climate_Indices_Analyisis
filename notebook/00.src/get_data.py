#!/usr/bin/env python3
"""
FastAPI untuk sistem data iklim dengan akses ke database TimescaleDB.
Mendukung:
- Query data QC, homogenisasi, extended
- Perhitungan anomali bulanan & tahunan suhu (TEMPERATURE_AVG_C)
- Filter berbasis kelengkapan data per stasiun dan per periode
Kompatibel dengan skema database:
  - observations.primary_key = (time, wmo_id, parameter, source, baseline)
  - baseline = NULL untuk raw/qc
  - baseline = '1981'/'1991' untuk homo/extended
"""

# from fastapi import FastAPI, Query, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
from sqlalchemy import create_engine, text
import yaml
from pathlib import Path
import logging
import calendar
import os

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = {'database': {
        'host': '172.19.0.201',
        'port': 5432,
        'name': 'climate_db',
        'user': 'api',
        'password': 'climate2026'
    }}
def get_db_engine(config_path=None):
    """Buat koneksi database SQLAlchemy."""
    url = f"postgresql://{config['database']['user']}:{config['database']['password']}@" \
          f"{config['database']['host']}:{config['database']['port']}/{config['database']['name']}"
    return create_engine(url)

# ==============================
# FUNGSI UTAMA QUERY DATA IKLIM
# ==============================
def query_climate_data(
    parameters=None,
    sources='qc',
    baseline=None,
    min_80pct_only=True,
    min_80pct_baseline='1991',
    start_date=None,
    end_date=None,
    stations=None,
    provinces=None,
    regions=None,
    time_aggregation=None,
    spatial_aggregation=None,
    limit=None,
    config_path=None
):
    engine = get_db_engine(config_path)
    
    if isinstance(sources, str):
        sources = [sources]
    valid_sources = {'raw', 'qc', 'homo', 'extended'}
    if not set(sources).issubset(valid_sources):
        raise ValueError(f"sources harus salah satu dari {valid_sources}")
    
    has_homo = any(s in sources for s in ['homo', 'extended'])
    
    if has_homo:
        if baseline is None:
            raise ValueError("Parameter 'baseline' wajib ditentukan saat menggunakan source 'homo' atau 'extended'")
        if baseline not in ['1981', '1991']:
            raise ValueError("baseline harus '1981' atau '1991' untuk data homogenisasi")
    else:
        baseline = None

    if parameters is None:
        parameters = ['TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C', 'RAINFALL_24H_MM']
    elif isinstance(parameters, str):
        parameters = [parameters]

    effective_start_date = start_date
    if min_80pct_only and min_80pct_baseline == '1991':
        if effective_start_date is None or effective_start_date < '1991-01-01':
            effective_start_date = '1991-01-01'

    query = """
    SELECT 
        o.time,
        o.wmo_id,
        o.parameter,
        o.source,
        o.value,
        o.baseline,
        o.region,
        m.name,
        m.latitude,
        m.longitude,
        m.province,
        m.regency,
        m.elevation,
        a_obs.availability,
        a_obs.meets_80pct
    FROM observations o
    JOIN station_metadata m ON o.wmo_id = m.wmo_id
    JOIN station_availability a_obs 
        ON o.wmo_id = a_obs.wmo_id 
        AND o.parameter = a_obs.parameter 
        AND COALESCE(o.baseline::TEXT, 'none') = COALESCE(a_obs.baseline::TEXT, 'none')
    """
    
    if min_80pct_only:
        query += f"""
        JOIN station_availability a_filter 
            ON o.wmo_id = a_filter.wmo_id 
            AND o.parameter = a_filter.parameter 
            AND a_filter.baseline = :min_80pct_baseline
        """
    
    query += " WHERE 1=1"
    params = {}
    query += " AND o.source = ANY(:sources)"
    params['sources'] = sources

    if has_homo:
        query += " AND o.baseline = :baseline"
        params['baseline'] = baseline
    else:
        query += " AND o.baseline IS NULL"

    query += " AND o.parameter = ANY(:parameters)"
    params['parameters'] = parameters

    if min_80pct_only:
        query += " AND a_filter.meets_80pct = TRUE"
        params['min_80pct_baseline'] = min_80pct_baseline

    if effective_start_date:
        query += " AND o.time >= :start_date"
        params['start_date'] = effective_start_date
    if end_date:
        query += " AND o.time <= :end_date"
        params['end_date'] = end_date

    if stations:
        query += " AND o.wmo_id = ANY(:stations)"
        params['stations'] = stations
    if provinces:
        query += " AND m.province = ANY(:provinces)"
        params['provinces'] = provinces
    if regions:
        query += " AND o.region = ANY(:regions)"
        params['regions'] = regions

    query += " ORDER BY o.time, o.wmo_id"
    if limit:
        query += " LIMIT :limit"
        params['limit'] = limit

    df = pd.read_sql(text(query), engine, params=params)
    
    # Agregasi waktu
    if time_aggregation and len(df) > 0:
        df['time'] = pd.to_datetime(df['time'])
        if time_aggregation == 'monthly':
            df['time'] = df['time'].dt.to_period('M').dt.start_time
        elif time_aggregation == 'yearly':
            df['time'] = df['time'].dt.to_period('Y').dt.start_time
        
        group_cols = [
            'time', 'wmo_id', 'parameter', 'source', 'baseline', 'region',
            'name', 'latitude', 'longitude', 'province', 'regency', 'elevation'
        ]
        numeric_cols = ['value', 'availability']
        agg_dict = {col: 'mean' for col in numeric_cols}
        df = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    # Agregasi spasial
    if spatial_aggregation and len(df) > 0:
        df['time'] = pd.to_datetime(df['time'])
        if spatial_aggregation == 'province':
            group_cols = ['time', 'parameter', 'source', 'province']
        elif spatial_aggregation == 'region':
            group_cols = ['time', 'parameter', 'source', 'region']
        elif spatial_aggregation == 'national':
            group_cols = ['time', 'parameter', 'source']
        else:
            group_cols = ['time', 'parameter', 'source']
        
        numeric_cols = ['value', 'availability']
        agg_dict = {col: ['mean', 'std', 'count'] for col in numeric_cols}
        df = df.groupby(group_cols).agg(agg_dict).round(4).reset_index()
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    
    logger.info(
        f"✅ Mengambil {len(df)} baris data "
        f"(sources={sources}, baseline={baseline}, "
        f"filter kelengkapan dari baseline={min_80pct_baseline}, "
        f"waktu ≥ {effective_start_date})"
    )
    return df
