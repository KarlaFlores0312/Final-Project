# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 16:00:04 2026

@author: kmflo
"""

import os 
import glob 
import xarray as xr 
import matplotlib.pyplot as plt 
from matplotlib.colors import BoundaryNorm
import numpy as np

# -------------------------------
# 1) DATA LOADING
# ------------------------------- 

def load_multifile_dataset(folder_path):
    """
    Loads all tas and pr .nc files inside a folder as one xarray Dataset.
    """
    files = sorted(glob.glob(os.path.join(folder_path, "*.nc")))

    if len(files) == 0:
        raise FileNotFoundError(f"No NetCDF files found in: {folder_path}")

    datasets = [xr.open_dataset(f) for f in files]
    ds = xr.concat(datasets, dim="time")
    return ds

def load_scenario_data(base_data_path, scenario):
    """
    Loads tas and pr datasets for a given scenario.

    scenario options:
    - "historical"
    - "ssp126"
    - "ssp585"
    """
    tas_folder = os.path.join(base_data_path, "tas", scenario)
    pr_folder = os.path.join(base_data_path, "pr", scenario)

    ds_tas = load_multifile_dataset(tas_folder)
    ds_pr = load_multifile_dataset(pr_folder)

    tas = ds_tas["tas"]
    pr = ds_pr["pr"]

    return tas, pr 

# --------------------------------------
# 2) PERIOD SELECTION
# --------------------------------------

def select_period(data, start_year, end_year):
    """
    This function extracts the data (pr and tas) that falls 
    withing the chosen time period.
    
    IMPORTANT: Time is saved in strings like "2031", not in full dates.
    So we slice time like this : slice("2031", "2060")
    """
    return data.sel(time=slice(str(start_year), str(end_year))) 

def mean_over_period(data, start_year, end_year):
    """
    This function takes climate data(tas or pr) and a time period as input 
    and returns the mean value of the data over that period for each grid cell.
    The mean will be used in the AI calculation later.
    """
    subset = select_period(data, start_year, end_year)
    return subset.mean("time") 


# -----------------------------------------------
# 3) CLIMATE INDEX (DE MARTONNE)
# -----------------------------------------------

def kelvin_to_celsius(tas_k): 
    """
    this function takes temperature in kelvin as input and returns it in celsius
    """
    return tas_k - 273.15


def compute_de_martonne_index(P_mm_year, T_celsius):
    """
    this function calculates the De Martonne 
    index with precipitation in mm/year and temperature in celsius
    """
    denom = T_celsius + 10

    # Mask invalid denominator (T <= -10°C): this mask is necessary to 
    #avoid negative temperatures that make the AI invalid (negative).
    denom = denom.where(denom > 0.1)  #this avoids near 0 values

    AI = P_mm_year / denom
    return AI

# ---------------------------------------
# 4) CLIMATE CLASSIFICATION
# ---------------------------------------
def classify_aridity(ai):
    """
    Returns climate class codes:

    0: Arid           (AI < 10)
    1: Semi-arid      (10 <= AI < 20)
    2: Mediterranean  (20 <= AI < 24)
    3: Semi-humid     (24 <= AI < 28)
    4: Humid          (28 <= AI < 35)
    5: Very Humid     (35 <= AI < 55) 
    6: Extremely Humid (AI >= 55)
    
    """
    classes = xr.full_like(ai, fill_value=np.nan)

    classes = classes.where(~(ai < 10), other=0)
    classes = classes.where(~((ai >= 10) & (ai < 20)), other=1)
    classes = classes.where(~((ai >= 20) & (ai < 24)), other=2)
    classes = classes.where(~((ai >= 24) & (ai < 28)), other=3) 
    classes = classes.where(~((ai >= 28) & (ai < 35)), other=4) 
    classes = classes.where(~((ai >= 35) & (ai < 55)), other=5) 
    classes = classes.where(~(ai >= 55), other= 6) 
     
    classes = classes.where(ai.notnull())
    return classes 
  
# ---------------------------------------------------------
# 5) FULL CLIMATE PIPELINE:climate calculations only (this will go in the main loop)
# ------------------------------------------------------------

def run_climate_pipeline(base_data_path, scenario, start_year, end_year,
                         hist_start=1981, hist_end=2010):
    """
    Runs full climate workflow and returns results in a dictionary.
    """

    # --- Load historical and future scenario
    tas_hist_all, pr_hist_all = load_scenario_data(base_data_path, "historical")
    tas_fut_all, pr_fut_all = load_scenario_data(base_data_path, scenario)

    # --- Calculates historical mean climate
    T_hist = kelvin_to_celsius(mean_over_period(tas_hist_all, hist_start, hist_end))
    P_hist = mean_over_period(pr_hist_all, hist_start, hist_end)

    # --- Calculates future mean climate
    T_fut = kelvin_to_celsius(mean_over_period(tas_fut_all, start_year, end_year))
    P_fut = mean_over_period(pr_fut_all, start_year, end_year) 
    
    # Mask only cells that are zero in BOTH periods (over ocean pr is 0)
    ocean_mask = (P_hist == 0) & (P_fut == 0)

    P_hist = P_hist.where(~ocean_mask)
    P_fut  = P_fut.where(~ocean_mask)

    T_hist = T_hist.where(~ocean_mask)
    T_fut  = T_fut.where(~ocean_mask)

    # --- Calculation of the De Martonne aridity index
    AI_hist = compute_de_martonne_index(P_hist, T_hist)
    AI_fut = compute_de_martonne_index(P_fut, T_fut)

    # --- Classification
    climate_hist = classify_aridity(AI_hist)
    climate_future = classify_aridity(AI_fut)


    return {
        "scenario": scenario,
        "future_period": (start_year, end_year),
        "historical_period": (hist_start, hist_end),

        "T_hist": T_hist,
        "P_hist": P_hist,
        "AI_hist": AI_hist,
        "climate_hist": climate_hist,

        "T_future": T_fut,
        "P_future": P_fut,
        "AI_future": AI_fut,
        "climate_future": climate_future,

        "lat": AI_hist["lat"],
        "lon": AI_hist["lon"],
    }

#------------------------------------------------------------
# 6) PLOTTING FUNCTION (this also goes in the main loop)
#-----------------------------------------------------------
def plot_climate_classification_geo(climate_map, title="Climate Classification Map"):
    data = climate_map.values.astype(float)
    lats = climate_map["lat"].values
    lons = climate_map["lon"].values

    class_names = [
        "Arid",
        "Semi-arid",
        "Mediterranean",
        "Semi-humid",
        "Humid",
        "Very Humid",
        "Extremely humid"
    ]

    n_classes = len(class_names)

    # Mask NaNs -> white
    masked = np.ma.masked_invalid(data)

    cmap = plt.cm.get_cmap("tab10", n_classes)
    cmap.set_bad("white")

    bounds = np.arange(-0.5, n_classes + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)

    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    plt.figure(figsize=(14, 6))
    img = plt.imshow(masked, origin="upper", extent=extent, aspect="auto",
                     cmap=cmap, norm=norm)

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    ticks = list(range(n_classes))
    cbar = plt.colorbar(img, ticks=ticks)
    cbar.set_ticklabels(class_names)
    cbar.set_label("Climate class")

    plt.tight_layout()
    plt.show()
