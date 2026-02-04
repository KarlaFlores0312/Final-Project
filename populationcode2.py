"""
Module to calculate population in each climate zone with possible resettlement

Key Features:
    
- Loads population data for year 2005 from NetCDF files
- Counts population distribution across climate zones
- Implements resettlement algorithm
- Generates tables and maps

"""

import os
import re
from typing import Any, Dict, Union

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# try to import scipy for fast spatial search (KDTree)
# KDTree enables efficient nearest-neighbor queries for resettlement
try:
    from scipy.spatial import cKDTree
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ============================================================
# 1) LOAD POPULATION DATA (YEAR 2005)
# ============================================================

TimeValue = Union[int, float, np.datetime64]


def _infer_time_value_for_year(pop_da: xr.DataArray, target_year: int = 2005) -> TimeValue:
   
    """
    Automatically infer how to select a specific year from the NetCDF time coordinate.
    
    Different NetCDF files encode time differently:
    - As datetime objects: np.datetime64("2005-01-01")
    - As offsets: "years since 1860" → 145 for year 2005
    - As plain years: 2005
    
    This function handles all common formats automatically.
    
    Args:
        pop_da: Population DataArray with time dimension
        target_year: Year to select (default: 2005)
    
    Returns:
        Time value that can be used in .sel(time=value)
   
    """
    time = pop_da["time"]

    # case 1: time is already decoded as datetime64
    if np.issubdtype(time.dtype, np.datetime64):
        return np.datetime64(f"{target_year}-01-01")

    # case 2: time has units like "years since 1860-1-1"
    # extract base year and calculate offset
    units = time.attrs.get("units", "")
    m = re.search(r"years\s+since\s+(\d{4})", str(units))
    if m:
        base_year = int(m.group(1))
        return target_year - base_year  # e.g., 2005 - 1860 = 145

    # case 3: fallback - assume time values are plain years
    return target_year


def load_population_2005(base_data_path: str) -> xr.DataArray:
    
    """
    Load population data for year 2005 from NetCDF file
    
    The function:
    - finds NetCDF files in the population folder
    - detects the population variable name
    - selects year 2005 using robust time selection
    - returns a 2D grid (latitude × longitude)
    
    Args:
        base_data_path: Root path to data directory
    
    Returns:
        xr.DataArray: Population grid for 2005 with shape (lat, lon)
    
    Raises:
        FileNotFoundError: If population folder or files not found
        ValueError: If time dimension is missing
    """
   
    print("\n" + "=" * 70)
    print("LOADING POPULATION DATA")
    print("=" * 70)

    # build path to population folder
    pop_folder = os.path.join(base_data_path, "population")
    print(f"Looking for population files in: {pop_folder}")

    if not os.path.exists(pop_folder):
        raise FileNotFoundError(f"Population folder not found: {pop_folder}")

    # find all NetCDF files
    files = sorted([f for f in os.listdir(pop_folder) if f.endswith(".nc") or f.endswith(".nc4")])
    print(f"Found {len(files)} NetCDF file(s)")
    if not files:
        raise FileNotFoundError("No population NetCDF file found in population folder.")

    # open first NetCDF file found
    print(f"Files found: {files}")
    pop_file = os.path.join(pop_folder, files[0])
    print(f"Opening file: {pop_file}")

    # decode_times=False prevents calendar-related errors with "years since" time encoding
    ds = xr.open_dataset(pop_file, decode_times=False)
    print("✓ File opened successfully")

    # detect population variable name
    # prefer "number_of_people" if it exists, otherwise use first variable
    if "number_of_people" in ds.data_vars:
        varname = "number_of_people"
    else:
        varname = list(ds.data_vars.keys())[0]
    print(f"Population variable name: '{varname}'")

    pop = ds[varname]
    print(f"Population data shape: {pop.shape}")
    print(f"Dimensions: {pop.dims}")

    # verify time dimension exists
    if "time" not in pop.dims:
        ds.close()
        raise ValueError("Population dataset does not contain a time dimension.")

    print(f"Time dimension found. Total time steps: {len(pop.time)}")
    print(f"Time units: {pop.time.attrs.get('units', '(none)')}")
    print(f"Time values (first 5): {pop.time.values[:5]}")
    print(f"Time values (last 5): {pop.time.values[-5:]}")

    # robust year selection with multiple fallback strategies
    target_year = 2005
    time_value = _infer_time_value_for_year(pop, target_year)

    # strategy 1: try selecting using inferred time value
    try:
        pop2005 = pop.sel(time=time_value)
        print(f"✓ Selected year {target_year} using time={time_value!r}")
    except Exception:
        # strategy 2: try selecting by string year
        try:
            pop2005 = pop.sel(time=str(target_year))
            print(f"✓ Selected year {target_year} using time='{target_year}'")
        except Exception:
            # strategy 3: fallback to last timestep (typically 2005 for 1861-2005 data)
            pop2005 = pop.isel(time=-1)
            print(f"⚠ Fell back to last timestep (index -1) for year {target_year}")

    # ensure result is 2D (lat, lon) by removing any remaining time dimension
    if "time" in pop2005.dims:
        pop2005 = pop2005.isel(time=0)

    # calculate and display total population
    total_pop = float(pop2005.sum(skipna=True).item())
    print("\n✓ Population data loaded successfully!")
    print(f"  Total global population in 2005: {total_pop:,.0f} people")
    print(f"  Data shape: {tuple(pop2005.shape)} (lat x lon)")
    print("=" * 70 + "\n")

    ds.close()
    return pop2005


# ============================================================
# 2) COUNT POPULATION BY CLIMATE ZONE
# ============================================================

def population_by_climate_zone(pop2005: xr.DataArray, climate_map: xr.DataArray, class_names: Dict[int, str]) -> pd.DataFrame:
    
    """
    Count total population living in each climate zone
    
    This function:
    1. For each climate class (0-6), creates a mask of matching cells
    2. Sums population only in those cells
    3. Returns results as a formatted pandas DataFrame
    4. Validates that total population is conserved
    
    Args:
        pop2005: Population grid (lat, lon)
        climate_map: Climate classification grid with values 0-6 (lat, lon)
        class_names: Mapping from climate codes to names
                    {0: "Arid", 1: "Semi-arid", ..., 6: "Extremely Humid"}
    
    Returns:
        pd.DataFrame with columns:
            - class_code: Climate class number (0-6)
            - class_name: Climate class name
            - population_2005: Total population in that zone
    """
   
    print("\n" + "=" * 70)
    print("COUNTING POPULATION PER CLIMATE ZONE")
    print("=" * 70)

    results = []
    
    # loop through each climate class
    for k, name in class_names.items():
        print(f"Processing {name} (class {k})...")
        
        # key operation: .where() creates mask, then sum population
        # example: climate_map == 0 creates boolean array [True, False, True, ...]
        #          .where() keeps values where True, sets rest to NaN
        #          .sum(skipna=True) sums only the valid (non-NaN) values
        total_pop = pop2005.where(climate_map == k).sum(skipna=True).item()
        
        print(f"  Population: {total_pop:,.0f}")
        results.append({"class_code": k, "class_name": name, "population_2005": float(total_pop)})

    df = pd.DataFrame(results)

    # validation: verify population conservation
    # sum across all zones should equal total population
    total_from_zones = float(df["population_2005"].sum())
    total_original = float(pop2005.sum(skipna=True).item())
    print(f"\n✓ Total from all zones: {total_from_zones:,.0f}")
    print(f"  Original total: {total_original:,.0f}")
    
    if np.isclose(total_from_zones, total_original, rtol=0.01):
        print("  ✓ Population conserved!")
    else:
        print("  ⚠ WARNING: Population totals don't match (check masks/NaNs/alignment).")
    
    print("=" * 70 + "\n")

    return df


# ============================================================
# 3) POPULATION DENSITY
# ============================================================

def simple_density(pop2005: xr.DataArray) -> xr.DataArray:
    
    """
    Simple population density calculation (placeholder)
    
    Currently just returns the population grid as-is.
    For true density (people/km²), necessary to divide by grid cell area,
    which varies with latitude.
    
    Args:
        pop2005: Population grid
    
    Returns:
        Same population grid (can be extended for true density calculation)
    """
    
    print("\n" + "=" * 70)
    print("CALCULATING POPULATION DENSITY")
    print("=" * 70)
    print("(Simple version: just returning population grid)")
    print("=" * 70 + "\n")
    return pop2005


# ============================================================
# 4) RESETTLEMENT ALGORITHM
# ============================================================

def _build_latlon_points(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    
    """
    Build array of all (latitude, longitude) coordinates for grid cells
    
    Creates a 2D meshgrid then flattens to (N, 2) array where N = nlat × nlon,
    each row contains [lat, lon] for one grid cell.
    
    Args:
        lat: 1D array of latitude values
        lon: 1D array of longitude values
    
    Returns:
        np.ndarray of shape (N, 2) with all (lat, lon) pairs
    """
   
    lon2d, lat2d = np.meshgrid(lon, lat)
    return np.column_stack([lat2d.ravel(), lon2d.ravel()])


def resettle_population(pop2005: xr.DataArray, climate_hist: xr.DataArray, climate_future: xr.DataArray) -> xr.DataArray:
    
    """
    Population resettlement algorithm for climate change adaptation
    
    Migration rules:
    1. Trigger: Cell must migrate if future_climate < historical_climate (drier)
    2. Destination constraint: future_climate(dest) ≥ historical_climate(source)
       → Destination's future must be at least as wet as source's historical climate
    3. Selection criteria (in order of priority):
       a) Minimize climate gap: future_climate(dest) - historical_climate(source)
          → Prefer destinations with climate closest to source's historical climate
       b) Among equal gaps, choose geographically nearest destination
    
    Algorithm:
    1. Identify all cells requiring migration (future < historical)
    2. For each migrating cell:
       - Try gap = 0: Search in climate class equal to historical
       - Try gap = 1: Search in climate class historical + 1
       - Continue increasing gap until candidates found
       - Use KDTree to find geographically nearest candidate
    3. Move population from source to selected destination
    4. Validate total population conservation
    
    
    Args:
        pop2005: Population grid for 2005 (lat, lon)
        climate_hist: Historical climate classification (lat, lon), values 0-6
        climate_future: Future climate classification (lat, lon), values 0-6
    
    Returns:
        xr.DataArray: New population grid after resettlement (lat, lon)
    
    Raises:
        ValueError: If grid shapes don't match
        ImportError: If scipy is not installed (required for KDTree)
    """
    
    print("\n" + "=" * 70)
    print("RESETTLEMENT CALCULATION")
    print("=" * 70)

    # validate that all grids have same shape
    if pop2005.shape != climate_hist.shape or pop2005.shape != climate_future.shape:
        raise ValueError(
            f"Shape mismatch: pop2005 {pop2005.shape}, hist {climate_hist.shape}, future {climate_future.shape}. "
            "Make sure all grids are aligned to the same lat/lon."
        )

    # verify scipy is available for fast spatial search
    if not _HAVE_SCIPY:
        raise ImportError(
            "scipy is required for fast resettlement (cKDTree). "
            "Install scipy or ask for the pure-numpy fallback."
        )

    # extract coordinates
    lat = pop2005["lat"].values
    lon = pop2005["lon"].values

    # convert to numpy arrays for faster computation
    pop = pop2005.values.astype(float)
    hist = climate_hist.values
    fut = climate_future.values

    nlat, nlon = pop.shape

    # flatten 2D grids to 1D arrays (easier to work with in loops)
    pop_f = pop.ravel()
    hist_f = hist.ravel()
    fut_f = fut.ravel()

    # build array of (lat, lon) coordinates for all cells
    pts = _build_latlon_points(lat, lon)

    # identify valid cells (no NaN values in any of the three grids)
    valid_cell = np.isfinite(pop_f) & np.isfinite(hist_f) & np.isfinite(fut_f)
    
    # cells with population > 0
    pop_positive = valid_cell & (pop_f > 0)

    # key step: identify cells requiring migration
    # migration needed if future climate is drier (lower class) than historical
    need_move = pop_positive & (fut_f < hist_f)

    moved_total = float(pop_f[need_move].sum())
    movers_count = int(np.count_nonzero(need_move))
    print(f"Cells requiring movement: {movers_count:,}")
    print(f"Total people to move: {moved_total:,.0f}")

    # initialize new population grid
    # start by copying original, then subtract movers from source cells
    new_pop_f = pop_f.copy()
    new_pop_f[need_move] -= pop_f[need_move]  # Empty source cells

    # precompute candidate destinations for each climate class
    # this speeds up the search by avoiding repeated filtering
    # candidates_by_class[c] = indices of all cells where future climate = c
    candidates_by_class = {c: np.where(valid_cell & (fut_f == c))[0] for c in range(0, 7)}
    
    # dictionary to store KDTree for each climate class (lazy construction)
    trees: Dict[int, Any] = {}

    def _tree_for_class(c: int):
       
        """
        Build or retrieve KDTree for a specific climate class
        
        KDTree enables fast nearest-neighbor queries in O(log N) time
        instead of O(N) for naive distance calculations
        
        Args:
            c: Climate class (0-6)
        
        Returns:
            Tuple of (KDTree, candidate_indices) or None if no candidates
        """
       
        if c not in trees:
            cand = candidates_by_class[c]
            if cand.size == 0:
                trees[c] = None
            else:
                # build KDTree from (lat, lon) points of candidate cells
                trees[c] = (cKDTree(pts[cand]), cand)
        return trees[c]

    # get indices of all cells requiring migration
    movers_idx = np.where(need_move)[0]

    # process migrants grouped by their historical climate class
    # this allows batch processing and optimization
    for hclass in range(0, 7):
        # select all migrants with this historical climate class
        group = movers_idx[hist_f[movers_idx] == hclass]
        if group.size == 0:
            continue  # no migrants in this climate class

        # for each migrant in this group
        for midx in group:
            assigned = None
            
            # try increasing climate gaps until a destination is found
            # gap = 0: Same climate as historical (best option)
            # gap = 1: One level wetter
            # gap = 2: Two levels wetter, etc.
            for gap in range(0, 7 - hclass):
                tclass = hclass + gap  # target climate class
                
                tree_pack = _tree_for_class(tclass)
                if tree_pack is None:
                    continue  # no candidates in this climate class
                
                tree, cand_idx = tree_pack
                
                # query KDTree for nearest neighbor
                # pts[midx] = (lat, lon) of migrant
                # returns: distance and index in cand_idx array
                _, nn = tree.query(pts[midx], k=1)
                
                # convert to global cell index
                assigned = int(cand_idx[int(nn)])
                break  # found destination, stop searching

            # fallback: if no suitable destination found (shouldn't happen),
            # keep population in place
            if assigned is None:
                assigned = int(midx)

            # add migrant's population to destination cell
            new_pop_f[assigned] += pop_f[midx]

    # reshape 1D array back to 2D grid and create xarray DataArray
    pop_after = xr.DataArray(
        new_pop_f.reshape((nlat, nlon)),
        coords=pop2005.coords,
        dims=pop2005.dims,
        name="population_after_resettlement",
    )

    # validation: verify population conservation
    # total population before and after must be equal
    before = float(np.nansum(pop_f))
    after = float(np.nansum(pop_after.values))
    print("\nConservation check:")
    print(f"  Before: {before:,.0f}")
    print(f"  After : {after:,.0f}")
    
    if np.isclose(before, after, rtol=1e-10, atol=1e-6):
        print("  ✓ Population conserved!")
    else:
        print("  ⚠ WARNING: Population not exactly conserved (check NaNs/masks).")

    print("=" * 70 + "\n")
    return pop_after


# ============================================================
# 5) FULL POPULATION PIPELINE
# ============================================================

def run_population_pipeline(base_data_path: str, climate_results: Dict[str, xr.DataArray]) -> Dict[str, Any]:
    
    """
    Execute complete population analysis pipeline
    
    This function:
    1. Load population data for 2005
    2. Count population by historical climate zones
    3. Count population by future climate zones (no migration)
    4. Perform resettlement algorithm
    5. Count population by future climate zones (after migration)
    
    Args:
        base_data_path: Root path to data directory
        climate_results: Dictionary containing:
            - "climate_hist": Historical climate classification
            - "climate_future": Future climate classification
    
    Returns:
        Dictionary containing:
            - "pop2005": Original population grid
            - "pop_before": Population grid before resettlement
            - "pop_after": Population grid after resettlement
            - "table_hist": Population by historical climate zones
            - "table_future_no_move": Population by future zones (no migration)
            - "table_future_after_move": Population by future zones (after migration)
    """
    
    print("\n" + "#" * 70)
    print("# STARTING POPULATION ANALYSIS PIPELINE")
    print("#" * 70 + "\n")

    # step 1: load population data
    pop2005 = load_population_2005(base_data_path)

    # step 2: extract climate classifications
    print("Extracting climate classifications...")
    climate_hist = climate_results["climate_hist"]
    climate_future = climate_results["climate_future"]
    print(f"  Historical climate shape: {climate_hist.shape}")
    print(f"  Future climate shape: {climate_future.shape}")

    # define climate zone names (0-6 indexing system)
    class_names = {
        0: "Arid",              # De Martonne Index < 10
        1: "Semi-arid",         # 10 ≤ IDM < 20
        2: "Mediterranean",     # 20 ≤ IDM < 24
        3: "Semi-humid",        # 24 ≤ IDM < 28
        4: "Humid",             # 28 ≤ IDM < 35
        5: "Very Humid",        # 35 ≤ IDM < 55
        6: "Extremely Humid",   # IDM ≥ 55
    }
    print(f"  Using {len(class_names)} climate classes (0-6 indexing)\n")

    # step 3: population by historical climate zones
    table_hist = population_by_climate_zone(pop2005, climate_hist, class_names)
    
    # step 4: population by future climate zones without migration
    # this shows what would happen if people don't move
    table_future_no_move = population_by_climate_zone(pop2005, climate_future, class_names)

    # step 5: perform resettlement
    pop_before = pop2005
    pop_after = resettle_population(pop2005, climate_hist, climate_future)

    # step 6: population by future climate zones after migration
    # this shows the redistribution after people have moved
    table_future_after_move = population_by_climate_zone(pop_after, climate_future, class_names)

    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETE")
    print("#" * 70 + "\n")

    return {
        "pop2005": pop2005,
        "pop_before": pop_before,
        "pop_after": pop_after,
        "table_hist": table_hist,
        "table_future_no_move": table_future_no_move,
        "table_future_after_move": table_future_after_move,
    }


# ============================================================
# 6) SHOW RESULTS
# ============================================================

def show_results(results: Dict[str, Any], title_prefix: str = "") -> None:
    
    """
    Creates:
    1. Console output with population tables
    2. Two population density maps (before/after resettlement)
    3. Bar chart comparing population distribution
    
    Args:
        results: Output dictionary from run_population_pipeline()
        title_prefix: Optional prefix for plot titles (e.g., "SSP126 2031-2060")
    """
   
    pref = f"{title_prefix} " if title_prefix else ""

    # display tables in console
    print("\n" + "=" * 70)
    print(f"{pref}HISTORICAL POPULATION BY CLIMATE ZONE")
    print("=" * 70)
    print(results["table_hist"])

    print("\n" + "=" * 70)
    print(f"{pref}FUTURE POPULATION BY CLIMATE ZONE (NO MOVE)")
    print("=" * 70)
    print(results["table_future_no_move"])

    print("\n" + "=" * 70)
    print(f"{pref}FUTURE POPULATION BY CLIMATE ZONE (AFTER RESETTLEMENT)")
    print("=" * 70)
    print(results["table_future_after_move"])
    print("=" * 70)

    # extract population grids for visualization
    pop_before = results["pop_before"]
    pop_after = results["pop_after"]

    lat = pop_before["lat"].values
    lon = pop_before["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    def _plot_map(da: xr.DataArray, label: str) -> None:
        
        """
        Create population density map using log scale
        
        Log scale (log10(pop + 1)) is used because:
        - Population varies by many orders of magnitude
        - Makes both sparse and dense regions visible
        - +1 avoids log(0) = -inf
        
        Ocean/unpopulated cells are shown as white.
        """
       
        data = da.values.astype(float)
        
        # mask ocean and unpopulated areas
        ocean_mask = (~np.isfinite(data)) | (data <= 0)
        data = data.copy()
        data[ocean_mask] = np.nan
        
        # apply log transformation for better visualization
        data_plot = np.log10(data + 1.0)
        
        # set up colormap (white for ocean)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad("white")

        plt.figure(figsize=(12, 5))
        plt.pcolormesh(lon2d, lat2d, data_plot, shading="auto", cmap=cmap)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"{pref}Population 2005 ({label}) – log10(pop + 1)")
        plt.colorbar(label="log10(pop + 1)")
        plt.tight_layout()

    # create maps before and after resettlement
    _plot_map(pop_before, "Before resettlement")
    _plot_map(pop_after, "After resettlement")

    # create comparison bar chart
    # chows population by climate zone: no move vs. after resettlement
    df_a = results["table_future_no_move"].copy().set_index("class_name")
    df_b = results["table_future_after_move"].copy().set_index("class_name")

    zones = list(df_a.index)
    x = np.arange(len(zones))
    w = 0.4  # bar width

    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, df_a["population_2005"].values, width=w, label="Future zones (no move)")
    plt.bar(x + w / 2, df_b["population_2005"].values, width=w, label="Future zones (after resettlement)")
    plt.xticks(x, zones, rotation=30, ha="right")
    plt.ylabel("Population (people)")
    plt.title(f"{pref}Population by FUTURE climate zone – no move vs after resettlement")
    plt.legend()
    plt.tight_layout()

    plt.show()