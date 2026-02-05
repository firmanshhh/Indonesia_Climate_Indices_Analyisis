import numpy as np
import pandas as pd

def idxTemp(df, tave, tmax, tmin, ref_start=1991, ref_end=2020):
    """
    Menghitung 20+ indeks ekstrem suhu harian berdasarkan definisi ETCCDI.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom 'time' (datetime) dan kolom suhu
    tave : str
        Nama kolom suhu rata-rata harian (°C), contoh: 'TEMPERATURE_AVG_C'
    tmax : str
        Nama kolom suhu maksimum harian (°C), contoh: 'TEMP_24H_TX_C'
    tmin : str
        Nama kolom suhu minimum harian (°C), contoh: 'TEMP_24H_TN_C'
    ref_start : int, optional (default=1991)
        Tahun awal periode baseline untuk perhitungan persentil (10th/90th)
    ref_end : int, optional (default=2020)
        Tahun akhir periode baseline untuk perhitungan persentil (10th/90th)
    
    Returns
    -------
    pd.DataFrame
        DataFrame indeks ekstrem suhu agregat tahunan dengan kolom:
        
        === INDEKS SUHU RATA-RATA (Tm) ===
        TMm   : Rata-rata suhu rata-rata tahunan (°C)
        TMx   : Maksimum suhu rata-rata tahunan (°C)
        TMn   : Minimum suhu rata-rata tahunan (°C)
        Tm10P : Persentase hari dengan Tave < 10th percentile baseline (%)
        Tm90P : Persentase hari dengan Tave > 90th percentile baseline (%)
        Tm10  : Jumlah absolut hari dengan Tave < 10th percentile (hari)
        Tm90  : Jumlah absolut hari dengan Tave > 90th percentile (hari)
        
        === INDEKS SUHU MAKSIMUM (Tx) ===
        TXm   : Rata-rata suhu maksimum tahunan (°C)
        TXx   : Maksimum suhu maksimum tahunan (°C)
        TXn   : Minimum suhu maksimum tahunan (°C)
        Tx10P : Persentase hari dengan Tmax < 10th percentile baseline (%)
        Tx90P : Persentase hari dengan Tmax > 90th percentile baseline (%)
        Tx10  : Jumlah absolut hari dengan Tmax < 10th percentile (hari)
        Tx90  : Jumlah absolut hari dengan Tmax > 90th percentile (hari)
        
        === INDEKS SUHU MINIMUM (Tn) ===
        TNm   : Rata-rata suhu minimum tahunan (°C)
        TNx   : Maksimum suhu minimum tahunan (°C)
        TNn   : Minimum suhu minimum tahunan (°C)
        Tn10P : Persentase hari dengan Tmin < 10th percentile baseline (%)
        Tn90P : Persentase hari dengan Tmin > 90th percentile baseline (%)
        Tn10  : Jumlah absolut hari dengan Tmin < 10th percentile (hari)
        Tn90  : Jumlah absolut hari dengan Tmin > 90th percentile (hari)
        
        === INDEKS VARIABILITAS ===
        DTR   : Rata-rata *Diurnal Temperature Range* (Tmax - Tmin) tahunan (°C)
        ETR   : *Extreme Temperature Range* (TXx - TNn) tahunan (°C)
        
        === INDEKS SPELL PANAS/DINGIN ===
        WSDI  : *Warm Spell Duration Index* - jumlah hari dalam spell ≥6 hari 
                berturut-turut dengan Tmax > 90th percentile baseline (hari)
        CSDI  : *Cold Spell Duration Index* - jumlah hari dalam spell ≥6 hari 
                berturut-turut dengan Tmin < 10th percentile baseline (hari)
    
    Notes
    -----
    1. Persentil (10th/90th) dihitung dari seluruh data harian pada periode baseline
       (bukan rata-rata bulanan) sesuai definisi ETCCDI.
    2. Spell detection menggunakan algoritma *consecutive days* dengan kriteria:
       - Minimal 6 hari berturut-turut memenuhi threshold
       - Hari dengan missing data dianggap memutus spell
    3. Semua perhitungan tahunan mengabaikan hari dengan missing data pada parameter terkait
    4. Nilai NaN dikembalikan jika seluruh data dalam satu tahun adalah missing
    
    References
    ----------
    ETCCDI (2023). "Expert Team on Climate Change Detection and Indices: 
    Definitions and Guidelines". https://etccdi.org/
    """

    df['time']  = pd.to_datetime(df['time'])
    df['YEAR']  = df['time'].dt.year
    df['MONTH'] = df['time'].dt.month
    df['DAY']   = df['time'].dt.day

    def perc_idx(df, var, ref_start, ref_end):
        ref_mask = (df['YEAR'] >= ref_start) & (df['YEAR'] <= ref_end)
        p10 = df.loc[ref_mask, var].quantile(0.10)
        p90 = df.loc[ref_mask, var].quantile(0.90)
        T10p = df.groupby('YEAR')[var].apply(
            lambda x: np.nan if x.isna().all() else (x < p10).sum() / len(x) * 100
        )
        T90p = df.groupby('YEAR')[var].apply(
            lambda x: np.nan if x.isna().all() else (x > p90).sum() / len(x) * 100
        )
        T10abs = df.groupby('YEAR')[var].apply(
            lambda x: np.nan if x.isna().all() else (x < p10).sum()
        )
        T90abs = df.groupby('YEAR')[var].apply(
            lambda x: np.nan if x.isna().all() else (x > p90).sum()
        )

        return T10p, T90p, T10abs, T90abs  # tambahkan return p10/p90

    # Hitung persentil global untuk WSDI/CSDI
    ref_mask = (df['YEAR'] >= ref_start) & (df['YEAR'] <= ref_end)
    p10_tmin = df.loc[ref_mask, tmin].quantile(0.10)
    p90_tmax = df.loc[ref_mask, tmax].quantile(0.90)

    # Buat kolom boolean untuk spell
    df = df.copy()
    df['_warm_spell'] = df[tmax] > p90_tmax
    df['_cold_spell'] = df[tmin] < p10_tmin

    # Fungsi bantu untuk hitung hari dalam spell ≥6
    def count_spell_days(series):
        if series.isna().all():
            return np.nan
        s = series.fillna(False).astype(bool)
        groups = (~s).cumsum()
        in_valid_spell = s.groupby(groups).transform(lambda g: g.sum() >= 6)
        return in_valid_spell.sum()

    # Hitung WSDI dan CSDI per tahun
    WSDI = df.groupby('YEAR')['_warm_spell'].apply(count_spell_days)
    CSDI = df.groupby('YEAR')['_cold_spell'].apply(count_spell_days)

    # Hitung indeks lainnya
    df["DTR"] = df[tmax] - df[tmin]
    Tm10P, Tm90P, Tm10, Tm90 = perc_idx(df, tave, ref_start, ref_end)
    Tn10P, Tn90P, Tn10, Tn90 = perc_idx(df, tmin, ref_start, ref_end)
    Tx10P, Tx90P, Tx10, Tx90 = perc_idx(df, tmax, ref_start, ref_end)

    TMm = df[tave].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.mean())
    TMx = df[tave].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.max())
    TMn = df[tave].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.min())

    TXm = df[tmax].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.mean())
    TXx = df[tmax].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.max())
    TXn = df[tmax].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.min())

    TNx = df[tmin].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.max())
    TNn = df[tmin].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.min())
    TNm = df[tmin].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.mean())

    DTR = df["DTR"].groupby(df['YEAR']).apply(lambda x: np.nan if x.isna().all() else x.mean())
    ETR = TXx - TNn
    INDEK_T = pd.DataFrame({
        'TMm'  : round(TMm, 3), 'TMx': round(TMx, 3), 'TMn': round(TMn, 3),
        'TXm'  : round(TXm, 3), 'TXx': round(TXx, 3), 'TXn': round(TXn, 3),
        'TNx'  : round(TNx, 3), 'TNn': round(TNn, 3), 'TNm': round(TNm, 3),
        'DTR'  : round(DTR, 3), 'ETR': round(ETR, 3),
        'Tm10P': round(Tm10P, 3), 'Tm90P': round(Tm90P, 3),
        'Tn10P': round(Tn10P, 3), 'Tn90P': round(Tn90P, 3),
        'Tx10P': round(Tx10P, 3), 'Tx90P': round(Tx90P, 3),
        'Tm10' : round(Tm10, 3), 'Tm90': round(Tm90, 3),
        'Tn10' : round(Tn10, 3), 'Tn90': round(Tn90, 3),
        'Tx10' : round(Tx10, 3), 'Tx90': round(Tx90, 3),
        'WSDI': WSDI, 'CSDI': CSDI  # <-- ditambahkan di sini
    })

    # Bersihkan kolom sementara
    df.drop(columns=['_warm_spell', '_cold_spell'], inplace=True, errors='ignore')
    return INDEK_T

def idxRain(df, ch, ref_start=1991, ref_end=2020, min_wet_days=30):
    """
    Menghitung 20+ indeks ekstrem curah hujan harian berdasarkan definisi ETCCDI
    dengan sistem QC robust untuk menangani keterbatasan data di wilayah tropis.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom 'time' (datetime) dan kolom curah hujan
    ch : str
        Nama kolom curah hujan harian (mm), contoh: 'RAINFALL_24H_MM'
    ref_start : int, optional (default=1991)
        Tahun awal periode baseline untuk perhitungan R95p/R99p
    ref_end : int, optional (default=2020)
        Tahun akhir periode baseline untuk perhitungan R95p/R99p
    min_wet_days : int, optional (default=30)
        Minimum hari basah (>1.0 mm) yang diperlukan pada periode baseline 
        untuk menghitung threshold R95p/R99p. Nilai default 30 hari sesuai 
        rekomendasi untuk wilayah dengan variasi musiman tinggi.
    
    Returns
    -------
    pd.DataFrame
        DataFrame indeks ekstrem curah hujan agregat tahunan dengan kolom:
        
        === INDEKS JUMLAH & INTENSITAS ===
        PRECTOT : Total curah hujan tahunan (mm)
        SDII    : *Simple Daily Intensity Index* - rata-rata curah hujan 
                  pada hari basah (≥1.0 mm) (mm/hari)
        
        === INDEKS FREKUENSI HARI BASAH ===
        HH      : Jumlah hari basah (curah hujan ≥1.0 mm) (hari)
        HH20MM  : Jumlah hari dengan curah hujan ≥20.0 mm (hari)
        HH50MM  : Jumlah hari dengan curah hujan ≥50.0 mm (hari)
        HH100MM : Jumlah hari dengan curah hujan ≥100.0 mm (hari)
        HH150MM : Jumlah hari dengan curah hujan ≥150.0 mm (hari)
        FH20    : Persentase hari ≥20mm terhadap total hari basah (%)
        FH50    : Persentase hari ≥50mm terhadap total hari basah (%)
        FH100   : Persentase hari ≥100mm terhadap total hari basah (%)
        FH150   : Persentase hari ≥150mm terhadap total hari basah (%)
        R50     : Alias untuk HH50MM (kompatibilitas dengan literatur lama)
        
        === INDEKS KEKERINGAN/KELEMBAPAN BERKEPANJANGAN ===
        CDD     : *Consecutive Dry Days* - maksimum hari kering berturut-turut 
                  (curah hujan <1.0 mm) (hari)
        CWD     : *Consecutive Wet Days* - maksimum hari basah berturut-turut 
                  (curah hujan ≥1.0 mm) (hari)
        
        === INDEKS CURAH HUJAN MAKSIMUM ===
        RX1DAY  : Curah hujan maksimum 1-hari (mm)
        RX5DAY  : Curah hujan maksimum 5-hari berturut-turut (mm)
        RX7DAY  : Curah hujan maksimum 7-hari berturut-turut (mm)
        RX10DAY : Curah hujan maksimum 10-hari berturut-turut (mm)
        
        === INDEKS EKSTREM PERSENTIL ===
        R95P    : Total curah hujan dari hari dengan curah hujan >95th percentile 
                  pada hari basah (>1mm) di baseline (mm)
        R99P    : Total curah hujan dari hari dengan curah hujan >99th percentile 
                  pada hari basah (>1mm) di baseline (mm)
        R95Ptot : Persentase R95P terhadap total curah hujan tahunan (%)
        R99Ptot : Persentase R99P terhadap total curah hujan tahunan (%)
        
        === METADATA QC (PENTING UNTUK TRACEABILITY) ===
        R95p_threshold_mm : Threshold 95th percentile yang digunakan (mm)
        R99p_threshold_mm : Threshold 99th percentile yang digunakan (mm)
        baseline_period   : Periode baseline yang efektif digunakan (str)
        qc_flag           : Kode kualitas data (lihat penjelasan di bawah)
    
    QC Flags yang Mungkin
    --------------------
    'OK'                                       : Baseline utama valid
    'BASELINE_1991_2020'                       : Menggunakan baseline 1991-2020
    'BASELINE_FALLBACK_1981_2010'              : Fallback ke 1981-2010
    'BASELINE_FALLBACK_FULL_PERIOD'            : Fallback ke seluruh periode data
    'NO_RAINFALL_DATA'                         : Tidak ada data curah hujan sama sekali
    'NO_WET_DAYS_IN_ENTIRE_RECORD'             : Tidak ada hari basah (>0mm) di seluruh record
    'INSUFFICIENT_WET_DAYS_{n}_OF_{m}_REQUIRED': Hari basah tidak cukup untuk threshold
    
    Notes
    -----
    1. Definisi "hari basah" untuk perhitungan persentil: curah hujan > 1.0 mm 
       (sesuai standar ETCCDI untuk wilayah tropis).
    2. Strategi fallback baseline:
       - Prioritas 1: 1991-2020 (default WMO)
       - Prioritas 2: 1981-2010 (jika prioritas 1 tidak memenuhi min_wet_days)
       - Prioritas 3: Seluruh periode data yang tersedia
       - Jika semua gagal: threshold diset NaN → R95P/R99P = NaN
    3. Perhitungan CDD/CWD:
       - Missing data dianggap memutus spell (conservative approach)
       - Threshold hari kering: <1.0 mm (sesuai ETCCDI)
    4. Untuk stasiun dengan data tidak lengkap (<80% availabilitas), disarankan 
       melakukan filter terlebih dahulu sebelum memanggil fungsi ini.
    
    References
    ----------
    Zhang, X. et al. (2011). "Indices for monitoring changes in extremes based on 
    daily temperature and precipitation data". WIREs Clim Change, 2: 851-870.
    ETCCDI (2023). "Guidelines on Climate Extremes Analysis". https://etccdi.org/
    """

    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df['YEAR'] = df['time'].dt.year
    
    # === VALIDASI 1: Tidak ada data curah hujan sama sekali ===
    if df[ch].isna().all() or df[ch].dropna().empty:
        years = sorted(df['YEAR'].unique())
        empty_result = _create_empty_rainfall_index(years, qc_flag='NO_RAINFALL_DATA')
        return empty_result
    
    # === VALIDASI 2: Tidak ada hari basah (>0) sama sekali ===
    all_valid = df[ch].dropna()
    if all_valid.empty or (all_valid <= 0).all():
        years = sorted(df['YEAR'].unique())
        empty_result = _create_empty_rainfall_index(years, qc_flag='NO_WET_DAYS_IN_ENTIRE_RECORD')
        return empty_result
    
    # === COBA PERIODE BASELINE UTAMA (1991-2020) ===
    ref_mask = (df['YEAR'] >= ref_start) & (df['YEAR'] <= ref_end)
    ref_data = df.loc[ref_mask, ch].dropna()
    wet_days_ref = ref_data[ref_data > 1.0]  # Hari basah (>1mm)
    
    # === FALLBACK STRATEGY ===
    baseline_used = f"{ref_start}-{ref_end}"
    qc_flag = "OK"
    R95p_threshold = R99p_threshold = np.nan
    
    if len(wet_days_ref) >= min_wet_days:
        # Baseline utama valid
        R95p_threshold = wet_days_ref.quantile(0.95)
        R99p_threshold = wet_days_ref.quantile(0.99)
        qc_flag = "BASELINE_1991_2020"
    else:
        # Coba baseline alternatif 1981-2010
        alt_start, alt_end = 1981, 2010
        alt_mask = (df['YEAR'] >= alt_start) & (df['YEAR'] <= alt_end)
        alt_data = df.loc[alt_mask, ch].dropna()
        alt_wet = alt_data[alt_data > 1.0]
        
        if len(alt_wet) >= min_wet_days:
            R95p_threshold = alt_wet.quantile(0.95)
            R99p_threshold = alt_wet.quantile(0.99)
            baseline_used = f"{alt_start}-{alt_end}"
            qc_flag = f"BASELINE_FALLBACK_1981_2010"
        else:
            # Coba gunakan seluruh periode data yang tersedia
            all_wet = all_valid[all_valid > 1.0]
            if len(all_wet) >= min_wet_days:
                R95p_threshold = all_wet.quantile(0.95)
                R99p_threshold = all_wet.quantile(0.99)
                min_yr, max_yr = df['YEAR'].min(), df['YEAR'].max()
                baseline_used = f"FULL_PERIOD_{min_yr}_{max_yr}"
                qc_flag = "BASELINE_FALLBACK_FULL_PERIOD"
            else:
                # Tidak ada cukup hari basah di manapun → set threshold ke NaN
                qc_flag = f"INSUFFICIENT_WET_DAYS_{len(wet_days_ref)}_OF_{min_wet_days}_REQUIRED"
                baseline_used = "NONE"
    
    # === HELPER FUNCTIONS (sama seperti implementasi sebelumnya) ===
    def is_all_nan(arr):
        return np.isnan(arr).all()
    
    def count_wet_days(arr, threshold=1.0):
        if is_all_nan(arr):
            return np.nan
        valid = arr[~np.isnan(arr)]
        return np.sum(valid >= threshold)
    
    def calc_cdd(arr, threshold=1.0):
        if is_all_nan(arr):
            return np.nan
        max_cdd = current = 0
        for val in arr:
            if np.isnan(val):
                current = 0
                continue
            if val < threshold:
                current += 1
                max_cdd = max(max_cdd, current)
            else:
                current = 0
        return max_cdd
    
    def calc_cwd(arr, threshold=1.0):
        if is_all_nan(arr):
            return np.nan
        max_cwd = current = 0
        for val in arr:
            if np.isnan(val):
                current = 0
                continue
            if val >= threshold:
                current += 1
                max_cwd = max(max_cwd, current)
            else:
                current = 0
        return max_cwd
    
    def calc_rxnday(arr, n_days=1):
        if is_all_nan(arr) or len(arr) < n_days:
            return np.nan
        max_sum = -np.inf
        found_valid = False
        for i in range(len(arr) - n_days + 1):
            window = arr[i:i+n_days]
            if np.isnan(window).any():
                continue
            window_sum = np.sum(window)
            if not np.isnan(window_sum):
                max_sum = max(max_sum, window_sum)
                found_valid = True
        return max_sum if found_valid else np.nan
    
    # === HITUNG R95p/R99p DENGAN PENANGANAN THRESHOLD NaN ===
    def calc_r95p(arr, threshold=R95p_threshold):
        if np.isnan(threshold) or is_all_nan(arr):
            return np.nan
        valid = arr[~np.isnan(arr)]
        wet = valid[valid > 1.0]
        if len(wet) == 0:
            return np.nan
        return np.sum(wet[wet > threshold])
    
    def calc_r99p(arr, threshold=R99p_threshold):
        if np.isnan(threshold) or is_all_nan(arr):
            return np.nan
        valid = arr[~np.isnan(arr)]
        wet = valid[valid > 1.0]
        if len(wet) == 0:
            return np.nan
        return np.sum(wet[wet > threshold])
    
    # === AGREGASI TAHUNAN ===
    yearly_groups = df.groupby('YEAR')[ch]
    
    PRECTOT = yearly_groups.apply(lambda x: np.nan if x.isna().all() else x.sum())
    HH = yearly_groups.apply(lambda x: count_wet_days(x.values, 1.0))
    HH20MM = yearly_groups.apply(lambda x: count_wet_days(x.values, 20.0))
    HH50MM = yearly_groups.apply(lambda x: count_wet_days(x.values, 50.0))
    HH100MM = yearly_groups.apply(lambda x: count_wet_days(x.values, 100.0))
    HH150MM = yearly_groups.apply(lambda x: count_wet_days(x.values, 150.0))
    
    def calc_fraction(numerator, denominator):
        result = np.full_like(numerator, np.nan, dtype=np.float64)
        mask = (denominator > 0) & (~np.isnan(denominator)) & (~np.isnan(numerator))
        result[mask] = (numerator[mask] / denominator[mask] * 100).round(2)
        return result
    
    FH20 = calc_fraction(HH20MM.values, HH.values)
    FH50 = calc_fraction(HH50MM.values, HH.values)
    FH100 = calc_fraction(HH100MM.values, HH.values)
    FH150 = calc_fraction(HH150MM.values, HH.values)
    
    CDD = yearly_groups.apply(lambda x: calc_cdd(x.values, 1.0))
    CWD = yearly_groups.apply(lambda x: calc_cwd(x.values, 1.0))
    
    SDII = yearly_groups.apply(lambda x: 
        np.nan if x.isna().all() else 
        (x[x >= 1.0].sum() / len(x[x >= 1.0])) if len(x[x >= 1.0]) > 0 else np.nan
    )
    
    RX1DAY = yearly_groups.apply(lambda x: calc_rxnday(x.values, 1))
    RX5DAY = yearly_groups.apply(lambda x: calc_rxnday(x.values, 5))
    RX7DAY = yearly_groups.apply(lambda x: calc_rxnday(x.values, 7))
    RX10DAY = yearly_groups.apply(lambda x: calc_rxnday(x.values, 10))
    
    R95P = yearly_groups.apply(lambda x: calc_r95p(x.values, R95p_threshold))
    R99P = yearly_groups.apply(lambda x: calc_r99p(x.values, R99p_threshold))
    
    R95Ptot = calc_fraction(R95P.values, PRECTOT.values)
    R99Ptot = calc_fraction(R99P.values, PRECTOT.values)
    
    # === OUTPUT DENGAN METADATA QC ===
    INDEK_CH = pd.DataFrame({
        'PRECTOT': PRECTOT.round(1),
        'HH': HH.round(1),
        'HH20MM': HH20MM.round(1),
        'HH50MM': HH50MM.round(1),
        'HH100MM': HH100MM.round(1),
        'HH150MM': HH150MM.round(1),
        'FH20': FH20,
        'FH50': FH50,
        'FH100': FH100,
        'FH150': FH150,
        'R50': HH50MM.round(1),
        'CDD': CDD.round(1),
        'CWD': CWD.round(1),
        'SDII': SDII.round(1),
        'RX1DAY': RX1DAY.round(1),
        'RX5DAY': RX5DAY.round(1),
        'RX7DAY': RX7DAY.round(1),
        'RX10DAY': RX10DAY.round(1),
        'R95P': R95P.round(1),
        'R99P': R99P.round(1),
        'R95Ptot': R95Ptot,
        'R99Ptot': R99Ptot,
        # Metadata QC untuk traceability
        'R95p_threshold_mm': R95p_threshold,
        'R99p_threshold_mm': R99p_threshold,
        'baseline_period': baseline_used,
        'qc_flag': qc_flag
    })
    
    INDEK_CH.replace([np.inf, -np.inf], np.nan, inplace=True)
    INDEK_CH.index.name = 'YEAR'
    return INDEK_CH


def _create_empty_rainfall_index(years, qc_flag):
    """Helper untuk membuat DataFrame kosong dengan struktur yang benar."""
    empty_df = pd.DataFrame(index=years, columns=[
        'PRECTOT', 'HH', 'HH20MM', 'HH50MM', 'HH100MM', 'HH150MM',
        'FH20', 'FH50', 'FH100', 'FH150', 'R50', 'CDD', 'CWD', 'SDII',
        'RX1DAY', 'RX5DAY', 'RX7DAY', 'RX10DAY', 'R95P', 'R99P', 'R95Ptot', 'R99Ptot',
        'R95p_threshold_mm', 'R99p_threshold_mm', 'baseline_period', 'qc_flag'
    ])
    empty_df[:] = np.nan
    empty_df['qc_flag'] = qc_flag
    empty_df['baseline_period'] = "NONE"
    empty_df.index.name = 'YEAR'
    return empty_df