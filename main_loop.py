# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 15:06:13 2026

@author: kmflo
"""

import os
import matplotlib.pyplot as plt

from climate_classification import run_climate_pipeline, plot_climate_classification_geo
from populationcode2 import run_population_pipeline, show_results


# ------------------------------------
# BASE DATA PATH
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# IMPORTANT:
# The data should be inside a folder like this: Final_project_python/data
# if the base data path doesn't work adapt the folder name on the code. 
base_data_path = os.path.join(BASE_DIR, "Final_project_python", "data")

if not os.path.exists(base_data_path):
    raise FileNotFoundError(f"Data folder not found: {base_data_path}")


# ------------------------------------------------
# USER INPUT FUNCTIONS
# ------------------------------------------------

def get_user_scenario():
    while True:
        print("\nChoose future scenario:")
        print("1 - SSP126")
        print("2 - SSP585")

        choice = input("Your choice: ")

        if choice == "1":
            return "ssp126"
        elif choice == "2":
            return "ssp585"
        else:
            print("Invalid choice. Please try again.")


def get_user_period():
    while True:
        print("\nChoose future period:")
        print("1 - Near future (2031–2060)")
        print("2 - Far future (2071–2100)")
        print("3 - Custom period (minimum 30 years)")

        choice = input("Your choice: ")

        if choice == "1":
            return 2031, 2060

        elif choice == "2":
            return 2071, 2100

        elif choice == "3":
            try:
                start = int(input("Start year: "))
                end = int(input("End year: "))

                if end - start + 1 >= 30:
                    return start, end
                else:
                    print("Period must be at least 30 years.")
            except ValueError:
                print("Please enter valid years.")

        else:
            print("Invalid choice. Please try again.")


def show_menu():
    print("\nMenu:")
    print("1 - Show HISTORICAL climate classification map")
    print("2 - Show FUTURE climate classification map")
    print("3 - Show population tables")
    print("4 - Show population density & resettlement maps")
    print("5 - Show ALL outputs")
    print("6 - Change scenario / period")
    print("7 - Exit")

    return input("Your choice: ")


# ============================================================
# MAIN PROGRAM LOOP
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("GLOBAL CLIMATE & POPULATION ANALYSIS PROGRAM")
    print("=" * 70)

    while True:
        # ----------------------------------------------------
        # STEP 1: USER INPUT
        # ----------------------------------------------------
        scenario = get_user_scenario()
        start_year, end_year = get_user_period()

        print("\nRunning climate pipeline...")
        climate_results = run_climate_pipeline(
            base_data_path=base_data_path,
            scenario=scenario,
            start_year=start_year,
            end_year=end_year
        )

        print("\nRunning population pipeline...")
        population_results = run_population_pipeline(
            base_data_path=base_data_path,
            climate_results=climate_results
        )

        # ----------------------------------------------------
        # STEP 2: MENU LOOP
        # ----------------------------------------------------
        while True:
            choice = show_menu()

            if choice == "1":
                plot_climate_classification_geo(
                    climate_results["climate_hist"],
                    title="Historical Climate Classification (1981–2010)"
                )

            elif choice == "2":
                plot_climate_classification_geo(
                    climate_results["climate_future"],
                    title=f"Future Climate Classification ({scenario.upper()}, {start_year}–{end_year})"
                )

            elif choice == "3":
                print("\nPopulation by climate zone (2005):")
                print(population_results["table_hist"])

                print("\nFuture population by climate zone (NO resettlement):")
                print(population_results["table_future_no_move"])

                print("\nFuture population by climate zone (AFTER resettlement):")
                print(population_results["table_future_after_move"])

            elif choice == "4":
                show_results(
                    population_results,
                    title_prefix=f"{scenario.upper()} {start_year}–{end_year}"
                )

            elif choice == "5":
                # Show climate maps first
                plot_climate_classification_geo(
                    climate_results["climate_hist"],
                    title="Historical Climate Classification (1981–2010)"
                )

                plot_climate_classification_geo(
                    climate_results["climate_future"],
                    title=f"Future Climate Classification ({scenario.upper()}, {start_year}–{end_year})"
                )

                print("\nClose climate figures to continue to population results...")
                plt.show()

                show_results(
                    population_results,
                    title_prefix=f"{scenario.upper()} {start_year}–{end_year}"
                )

            elif choice == "6":
                print("\nReturning to scenario and period selection...")
                break

            elif choice == "7":
                print("\nExiting program. Goodbye!")
                return

            else:
                print("Invalid option. Please try again.")


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()