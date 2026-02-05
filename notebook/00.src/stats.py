import numpy as np
from scipy.stats import linregress  # ✅ Import benar untuk analisis tren
import pandas as pd

def trend_analysis(df, min_data_points=5):
    """
    Analisis tren robust dengan penanganan index non-numerik dan kompatibilitas scipy.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame dengan index tahun (string/int) dan kolom numerik
    min_data_points : int
        Minimal titik data valid yang diperlukan (default: 5)
    
    Returns:
    --------
    pd.DataFrame dengan index = statistik ('slope', 'intercept', ...), kolom = variabel iklim
    """
    trends = {}
    
    # === PERBAIKAN KRITIS 1: Konversi index ke tahun numerik yang benar ===
    try:
        # Handle berbagai format index: string "1984", integer 1984, atau datetime
        if isinstance(df.index, pd.DatetimeIndex):
            years = df.index.year.astype(float).values
        elif pd.api.types.is_string_dtype(df.index) or pd.api.types.is_object_dtype(df.index):
            # Ekstrak 4 digit pertama sebagai tahun (handle "1984-01", "1984", dll)
            years = df.index.astype(str).str[:4].astype(int).astype(float).values
        else:
            years = df.index.astype(float).values
    except Exception as e:
        print(f"⚠️  Gagal konversi index ke tahun numerik: {e}. Menggunakan indeks urutan sebagai fallback.")
        years = np.arange(len(df), dtype=float)  # Fallback: 0, 1, 2, ...
    
    for col in df.columns:
        # Konversi nilai ke float dengan penanganan error
        try:
            values = pd.to_numeric(df[col], errors='coerce').values
        except Exception:
            values = np.full(len(df), np.nan)
        
        # Filter nilai valid (bukan NaN) untuk tahun dan nilai
        mask = (~np.isnan(values)) & (~np.isnan(years))
        valid_years = years[mask]
        valid_values = values[mask]
        n_valid = len(valid_values)
        
        # Tidak cukup data untuk analisis tren
        if n_valid < min_data_points:
            trends[col] = {
                'slope': np.nan, 'intercept': np.nan, 'r_value': np.nan,
                'p_value': np.nan, 'std_err': np.nan, 'n_valid': n_valid
            }
            continue
        
        # === PERBAIKAN KRITIS 2: Validasi variance > 0 (hindari pembagian nol) ===
        if n_valid < 2 or np.isclose(np.std(valid_values), 0, atol=1e-10):
            trends[col] = {
                'slope': 0.0,
                'intercept': np.mean(valid_values) if n_valid > 0 else np.nan,
                'r_value': 0.0,
                'p_value': 1.0,
                'std_err': 0.0,
                'n_valid': n_valid
            }
            continue
        
        # === PERBAIKAN KRITIS 3: Kompatibilitas scipy.linregress (tuple vs object) ===
        try:
            result = linregress(valid_years, valid_values)
            
            # Handle kompatibilitas scipy versi lama (tuple) dan baru (object)
            if isinstance(result, tuple) and len(result) == 5:
                # scipy < 1.9.0: returns tuple (slope, intercept, rvalue, pvalue, stderr)
                slope, intercept, r_value, p_value, std_err = result
            else:
                # scipy >= 1.9.0: returns LinregressResult object
                slope = result.slope
                intercept = result.intercept
                r_value = result.rvalue
                p_value = result.pvalue
                std_err = result.stderr
            
            # Validasi hasil numerik (hindari inf/nan dari komputasi)
            if not np.isfinite(slope):
                slope = np.nan
            if not np.isfinite(intercept):
                intercept = np.nan
            if not np.isfinite(r_value):
                r_value = np.nan
            if not np.isfinite(p_value):
                p_value = np.nan
            if not np.isfinite(std_err):
                std_err = np.nan
                
        except Exception as e:
            # Debug info untuk troubleshooting
            print(f"  ⚠️  linregress gagal untuk kolom '{col}': {str(e)[:80]}")
            print(f"      Data valid: {n_valid} titik, std={np.std(valid_values):.4f}, range=[{valid_values.min():.2f}, {valid_values.max():.2f}]")
            slope = intercept = r_value = p_value = std_err = np.nan
        
        trends[col] = {
            'slope': slope,
            'intercept': intercept,
            'r_value': r_value,
            'p_value': p_value,
            'std_err': std_err,
            'n_valid': n_valid
        }
    
    return pd.DataFrame(trends).T