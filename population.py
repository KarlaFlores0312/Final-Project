"""
- Loads population (2005) from the provided NetCDF (0.5° annual 1861-2005)
- Computes population by climate zone
- Implements resettlement rule:
    * If future climate is drier than historical climate in a cell, that cell's population must move
    * Population may only move to cells whose FUTURE climate is not drier than the cell's HISTORICAL climate
    * Choose destination that is as close as possible to historical climate (minimize climate gap),
      and among those, choose geographically nearest cell.

- Spyder-friendly display:
    * Shows result tables in the console (and keeps DataFrames for the Variable Explorer)
    * Shows plots interactively via matplotlib (no CSV/PNG files are created)
"""

import os
import re
from typing import Any, Dict, Union

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

# Optional, used for fast nearest-neighbor search.
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ============================================================
# 1) LOAD POPULATION DATA (YEAR 2005)
# ============================================================

TimeValue = Union[int, float, np.datetime64]


def _infer_time_value_for_year(pop_da: xr.DataArray, target_year: int = 2005) -> TimeValue:
    """
    Infer the coordinate value that corresponds to `target_year` for common NetCDF time encodings.
    Works for:
      - numeric time with units like "years since 1860-1-1 12:00:00"
      - datetime-like time coordinates
    """
    time = pop_da["time"]

    # If time already decoded as datetime
    if np.issubdtype(time.dtype, np.datetime64):
        return np.datetime64(f"{target_year}-01-01")

    units = time.attrs.get("units", "")
    # Example: "years since 1860-1-1 12:00:00"
    m = re.search(r"years\s+since\s+(\d{4})", str(units))
    if m:
        base_year = int(m.group(1))
        return target_year - base_year

    # Fallback: assume it is plain year values (e.g., 1861..2005)
    return target_year


def load_population_2005(base_data_path: str) -> xr.DataArray:
    """
    Loads the population dataset and returns population for the year 2005
    as a 2D grid (lat, lon).
    Expects a folder: <base_data_path>/population/ with a .nc or .nc4 file.
    """
    print("\n" + "=" * 70)
    print("LOADING POPULATION DATA")
    print("=" * 70)

    pop_folder = os.path.join(base_data_path, "population")
    print(f"Looking for population files in: {pop_folder}")

    if not os.path.exists(pop_folder):
        raise FileNotFoundError(f"Population folder not found: {pop_folder}")

    files = sorted([f for f in os.listdir(pop_folder) if f.endswith(".nc") or f.endswith(".nc4")])
    print(f"Found {len(files)} NetCDF file(s)")
    if not files:
        raise FileNotFoundError("No population NetCDF file found in population folder.")

    print(f"Files found: {files}")
    pop_file = os.path.join(pop_folder, files[0])
    print(f"Opening file: {pop_file}")

    # decode_times=False avoids calendar decoding issues with "years since ..."
    ds = xr.open_dataset(pop_file, decode_times=False)
    print("✓ File opened successfully")

    # Prefer the known variable name; otherwise fallback to first data_var
    if "number_of_people" in ds.data_vars:
        varname = "number_of_people"
    else:
        varname = list(ds.data_vars.keys())[0]
    print(f"Population variable name: '{varname}'")

    pop = ds[varname]
    print(f"Population data shape: {pop.shape}")
    print(f"Dimensions: {pop.dims}")

    if "time" not in pop.dims:
        ds.close()
        raise ValueError("Population dataset does not contain a time dimension.")

    print(f"Time dimension found. Total time steps: {len(pop.time)}")
    print(f"Time units: {pop.time.attrs.get('units', '(none)')}")
    print(f"Time values (first 5): {pop.time.values[:5]}")
    print(f"Time values (last 5): {pop.time.values[-5:]}")

    # Robust selection by year
    target_year = 2005
    time_value = _infer_time_value_for_year(pop, target_year)

    # Try select by computed time value
    try:
        pop2005 = pop.sel(time=time_value)
        print(f"✓ Selected year {target_year} using time={time_value!r}")
    except Exception:
        # Try selecting by string year if time values are strings
        try:
            pop2005 = pop.sel(time=str(target_year))
            print(f"✓ Selected year {target_year} using time='{target_year}'")
        except Exception:
            # Fallback: pick last timestep (typical for 1861-2005)
            pop2005 = pop.isel(time=-1)
            print(f"⚠ Fell back to last timestep (index -1) for year {target_year}")

    # Ensure 2D lat/lon
    if "time" in pop2005.dims:
        pop2005 = pop2005.isel(time=0)

    # Use skipna to be safe
    total_pop = float(pop2005.sum(skipna=True).item())
    print("\n✓ Population data loaded successfully!")
    print(f"  Total global population in 2005: {total_pop:,.0f} people")
    print(f"  Data shape: {tuple(pop2005.shape)} (lat x lon)")
    print("=" * 70 + "\n")

    ds.close()
    return pop2005


# ============================================================
# 2) POPULATION BY CLIMATE ZONE (TABLE)
# ============================================================

def population_by_climate_zone(pop2005: xr.DataArray, climate_map: xr.DataArray, class_names: Dict[int, str]) -> pd.DataFrame:
    """
    Returns a Pandas table with total population per climate zone.

    pop2005: xarray.DataArray (lat, lon)
    climate_map: xarray.DataArray (lat, lon) containing class codes (0..6)
    class_names: dict like {0:"Arid", 1:"Semi-arid", ...}
    """
    print("\n" + "=" * 70)
    print("COUNTING POPULATION PER CLIMATE ZONE")
    print("=" * 70)

    results = []
    for k, name in class_names.items():
        print(f"Processing {name} (class {k})...")
        total_pop = pop2005.where(climate_map == k).sum(skipna=True).item()
        print(f"  Population: {total_pop:,.0f}")
        results.append({"class_code": k, "class_name": name, "population_2005": float(total_pop)})

    df = pd.DataFrame(results)

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
# 3) POPULATION DENSITY (OPTIONAL SIMPLE VERSION)
# ============================================================

def simple_density(pop2005: xr.DataArray) -> xr.DataArray:
    """
    Simplest density-like map (not true people/km²).
    Just returns the population grid.
    """
    print("\n" + "=" * 70)
    print("CALCULATING POPULATION DENSITY")
    print("=" * 70)
    print("(Simple version: just returning population grid)")
    print("=" * 70 + "\n")
    return pop2005


# ============================================================
# 4) RESETTLEMENT (IMPLEMENTED)
# ============================================================

def _build_latlon_points(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return (N,2) points for all grid cell centers in (lat, lon) order."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    return np.column_stack([lat2d.ravel(), lon2d.ravel()])


def resettle_population(pop2005: xr.DataArray, climate_hist: xr.DataArray, climate_future: xr.DataArray) -> xr.DataArray:
    """
    Relocate population from cells that become drier (future < historical) to the nearest valid destination.

    Rules implemented:
      - If a cell's future class is drier than its historical class => must move.
      - Destination must satisfy: future_class(destination) >= historical_class(source)
      - Choose destination climate as close as possible to historical climate:
        minimize (future_class(destination) - historical_class(source)) (gap),
        then among those choose nearest geographic distance.

    Returns:
      pop_after: xarray.DataArray (lat, lon) with conserved total population.
    """
    print("\n" + "=" * 70)
    print("RESETTLEMENT CALCULATION")
    print("=" * 70)

    if pop2005.shape != climate_hist.shape or pop2005.shape != climate_future.shape:
        raise ValueError(
            f"Shape mismatch: pop2005 {pop2005.shape}, hist {climate_hist.shape}, future {climate_future.shape}. "
            "Make sure all grids are aligned to the same lat/lon."
        )

    if not _HAVE_SCIPY:
        raise ImportError(
            "scipy is required for fast resettlement (cKDTree). "
            "Install scipy or ask for the pure-numpy fallback."
        )

    lat = pop2005["lat"].values
    lon = pop2005["lon"].values

    pop = pop2005.values.astype(float)
    hist = climate_hist.values
    fut = climate_future.values

    nlat, nlon = pop.shape

    pop_f = pop.ravel()
    hist_f = hist.ravel()
    fut_f = fut.ravel()

    pts = _build_latlon_points(lat, lon)

    valid_cell = np.isfinite(pop_f) & np.isfinite(hist_f) & np.isfinite(fut_f)
    pop_positive = valid_cell & (pop_f > 0)

    need_move = pop_positive & (fut_f < hist_f)

    moved_total = float(pop_f[need_move].sum())
    movers_count = int(np.count_nonzero(need_move))
    print(f"Cells requiring movement: {movers_count:,}")
    print(f"Total people to move: {moved_total:,.0f}")

    new_pop_f = pop_f.copy()
    new_pop_f[need_move] -= pop_f[need_move]

    candidates_by_class = {c: np.where(valid_cell & (fut_f == c))[0] for c in range(0, 7)}
    trees: Dict[int, Any] = {}

    def _tree_for_class(c: int):
        if c not in trees:
            cand = candidates_by_class[c]
            if cand.size == 0:
                trees[c] = None
            else:
                trees[c] = (cKDTree(pts[cand]), cand)
        return trees[c]

    movers_idx = np.where(need_move)[0]

    for hclass in range(0, 7):
        group = movers_idx[hist_f[movers_idx] == hclass]
        if group.size == 0:
            continue

        for midx in group:
            assigned = None
            for gap in range(0, 7 - hclass):
                tclass = hclass + gap
                tree_pack = _tree_for_class(tclass)
                if tree_pack is None:
                    continue
                tree, cand_idx = tree_pack
                _, nn = tree.query(pts[midx], k=1)
                assigned = int(cand_idx[int(nn)])
                break

            if assigned is None:
                assigned = int(midx)

            new_pop_f[assigned] += pop_f[midx]

    pop_after = xr.DataArray(
        new_pop_f.reshape((nlat, nlon)),
        coords=pop2005.coords,
        dims=pop2005.dims,
        name="population_after_resettlement",
    )

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
    climate_results should be the dictionary returned by your climate pipeline:
      climate_results["climate_hist"], climate_results["climate_future"], etc.
    """
    print("\n" + "#" * 70)
    print("# STARTING POPULATION ANALYSIS PIPELINE")
    print("#" * 70 + "\n")

    pop2005 = load_population_2005(base_data_path)

    print("Extracting climate classifications...")
    climate_hist = climate_results["climate_hist"]
    climate_future = climate_results["climate_future"]
    print(f"  Historical climate shape: {climate_hist.shape}")
    print(f"  Future climate shape: {climate_future.shape}")

    class_names = {
        0: "Arid",
        1: "Semi-arid",
        2: "Mediterranean",
        3: "Semi-humid",
        4: "Humid",
        5: "Very Humid",
        6: "Extremely Humid",
    }
    print(f"  Using {len(class_names)} climate classes (0-6 indexing)\n")

    table_hist = population_by_climate_zone(pop2005, climate_hist, class_names)
    table_future_no_move = population_by_climate_zone(pop2005, climate_future, class_names)

    pop_before = pop2005
    pop_after = resettle_population(pop2005, climate_hist, climate_future)

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
# 6) SPYDER DISPLAY (TABLES + PLOTS; NO FILE OUTPUTS)
# ============================================================

def show_results_in_spyder(results: Dict[str, Any], title_prefix: str = "") -> None:
    """
    Show tables in console and plots interactively (no file outputs).
    Intended for Spyder; DataFrames remain accessible in the Variable Explorer.
    """
    pref = f"{title_prefix} " if title_prefix else ""

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

    pop_before = results["pop_before"]
    pop_after = results["pop_after"]

    lat = pop_before["lat"].values
    lon = pop_before["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    def _plot_map(da: xr.DataArray, label: str) -> None:
        data = da.values.astype(float)
        data_plot = np.log10(np.maximum(data, 0) + 1.0)

        plt.figure(figsize=(12, 5))
        plt.pcolormesh(lon2d, lat2d, data_plot, shading="auto")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"{pref}Population 2005 ({label}) – log10(pop + 1)")
        plt.colorbar(label="log10(pop + 1)")
        plt.tight_layout()

    _plot_map(pop_before, "Before resettlement")
    _plot_map(pop_after, "After resettlement")

    df_a = results["table_future_no_move"].copy().set_index("class_name")
    df_b = results["table_future_after_move"].copy().set_index("class_name")

    zones = list(df_a.index)
    x = np.arange(len(zones))
    w = 0.4

    plt.figure(figsize=(12, 5))
    plt.bar(x - w / 2, df_a["population_2005"].values, width=w, label="Future zones (no move)")
    plt.bar(x + w / 2, df_b["population_2005"].values, width=w, label="Future zones (after resettlement)")
    plt.xticks(x, zones, rotation=30, ha="right")
    plt.ylabel("Population (people)")
    plt.title(f"{pref}Population by FUTURE climate zone – no move vs after resettlement")
    plt.legend()
    plt.tight_layout()

    plt.show()


# ============================================================
# TEST CODE
# ============================================================

if __name__ == "__main__":
    print("\n" + "*" * 70)
    print("* POPULATION ANALYSIS MODULE - TEST MODE")
    print("*" * 70 + "\n")

    base_data_path = r"C:\Users\49152\Desktop\Final-Project\data"
    print(f"Base data path: {base_data_path}")
    print(f"Checking if path exists: {os.path.exists(base_data_path)}")

    if not os.path.exists(base_data_path):
        print("\n" + "!" * 70)
        print("! ERROR: base_data_path does not exist!")
        print("!" * 70)
        print("\nPlease update the base_data_path variable to point to your data folder.")
        print("Current path: " + base_data_path)
        print("\nExample paths to try:")
        print(r'  base_data_path = r"C:\Users\...\Final-Project"')
        print(r'  base_data_path = r"C:\Users\...\Final-Project\data"')
        print("\n" + "!" * 70 + "\n")
        raise SystemExit(1)

    try:
        print("\nTesting population data loading...")
        pop2005 = load_population_2005(base_data_path)

       

        climate_results = climate_module_a

        results = run_population_pipeline(base_data_path, climate_results)

        # Show tables + plots interactively (Spyder-friendly)
        show_results_in_spyder(results, title_prefix="TEST")

        print("\n✓ TEST COMPLETED SUCCESSFULLY!")

    except FileNotFoundError as e:
        print("\n" + "!" * 70)
        print("! FILE NOT FOUND ERROR")
        print("!" * 70)
        print(f"Error: {e}")
        print("\nPlease check:")
        print("1. Is your base_data_path correct?")
        print("2. Does the 'population' subfolder exist?")
        print("3. Is the population NetCDF file in that folder?")
        print("!" * 70 + "\n")
        raise

    except Exception as e:
        print("\n" + "!" * 70)
        print("! ERROR OCCURRED")
        print("!" * 70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("!" * 70 + "\n")
        raise
