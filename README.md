## 🌏 Overview

A Python script for analyzing trends in extreme rainfall and temperature events across Indonesia. The analysis is performed on rainfall and air temperature datasets from all observation stations of the Indonesian Agency for Meteorology, Climatology, and Geophysics (BMKG).

This script implements extreme index calculation methods based on the **ETCCDI (Expert Team on Climate Change Detection and Indices)** criteria. Additionally, it incorporates new indices specifically tailored to the climatic conditions of Indonesia. This project aims to support climate change adaptation and mitigation efforts in the country.

## 🔑 Key Features
- ✅ **47 ETCCDI based indices** for rainfall (e.g., CWD, CDD, PRCPTOT) and temperature (e.g., TXx, TNn, DTR, ETR)
- ✅ **National coverage**: All BMKG stations with quality-controlled and homogenized daily data (1981–present)
- ✅ **Multi-scale analysis**: Station-level, provincial/regional aggregation, and national summaries
- ✅ **Trend detection**: Mann-Kendall test + Sen's slope estimation with significance testing
