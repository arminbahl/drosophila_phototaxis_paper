import pandas as pd
from pathlib import Path
import numpy as np

root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")
#df = pd.read_hdf(root_path / "all_data_model_profile1.h5", key="raw_data")
df = pd.read_hdf(root_path / "all_data_model_profile2.h5", key="raw_data")
#df = pd.read_hdf(root_path / "all_data_deepposekit.h5", key="raw_data")
#df = pd.read_hdf(root_path / "all_data_spatial_phototaxis.h5", key="raw_data")

# Exclude fish
# df = df.query("larva_ID != '2018_02_15_fish012_setup0' and "
#               "larva_ID != '2018_02_15_fish015_setup1' and "
#               "larva_ID != '2018_02_15_fish016_setup1' and "
#               "larva_ID != '2018_02_15_fish017_setup0' and "
#               "larva_ID != '2018_02_15_fish020_setup0' and "
#               "larva_ID != '2018_02_15_fish021_setup1' and "
#               "larva_ID != '2018_02_15_fish022_setup1' and "
#               "larva_ID != '2018_02_15_fish024_setup1' and "
#               "larva_ID != '2018_03_01_fish030_setup1' and "
#               "larva_ID != '2018_11_20_fish036_setup0' and "
#               "larva_ID != '2018_11_27_fish037_setup0' and "
#               "larva_ID != '2018_11_27_fish006_setup1' and "
#               "larva_ID != '2018_02_15_fish023_setup1' and "
#               "larva_ID != '2018_02_15_fish018_setup0' and "
#               "larva_ID != '2018_02_15_fish014_setup1' and "
#               "larva_ID != '2018_03_01_fish027_setup0' and "
#               "larva_ID != '2018_03_12_fish040_setup1' and "
#               "larva_ID != '2018_11_20_fish035_setup0'")

df = df.query("larva_ID != '2018_02_15_fish012_setup0' and "
              "larva_ID != '2018_02_15_fish015_setup1' and "
              "larva_ID != '2018_02_15_fish016_setup1' and "
              "larva_ID != '2018_02_15_fish017_setup0' and "
              "larva_ID != '2018_02_15_fish020_setup0' and "
              "larva_ID != '2018_02_15_fish021_setup1' and "
              "larva_ID != '2018_02_15_fish022_setup1' and "
              "larva_ID != '2018_02_15_fish024_setup1' and "
              "larva_ID != '2018_03_01_fish030_setup1' and "
              "larva_ID != '2018_11_20_fish036_setup0' and "
              "larva_ID != '2018_11_27_fish037_setup0' and "
              "larva_ID != '2018_11_27_fish006_setup1' and "
              "larva_ID != '2018_02_15_fish023_setup1' and "
              "larva_ID != '2018_02_15_fish018_setup0' and "
              "larva_ID != '2018_02_15_fish014_setup1' and "
              "larva_ID != '2018_03_01_fish027_setup0' and "
              "larva_ID != '2018_03_12_fish040_setup1' and "
              "larva_ID != '2018_11_20_fish035_setup0' and "
              "larva_ID != '2018_11_27_fish013_setup1' and "
              "larva_ID != '2018_11_27_fish016_setup1'")


# Explusion for spatial_phototaxis
# df = df.query("larva_ID != '2017_12_11_fish002_setup1' and "
#               "larva_ID != '2017_12_11_fish003_setup0' and "
#               "larva_ID != '2017_12_11_fish005_setup1' and "
#               "larva_ID != '2017_12_11_fish006_setup1'")


all_results = dict({"experiment_name": [],
                    "larva_ID": [],
                    "region_bin": [],
                    "time_bin": [],
                    "fraction_of_time_spent": [],
                    "speed": []})

experiment_names = df.index.get_level_values('experiment_name').unique().values
for experiment_name in experiment_names:

    df_selected = df.query("experiment_name == @experiment_name").reset_index(level=['experiment_name'], drop=True)
    larva_IDs = df_selected.index.get_level_values('larva_ID').unique().values
    for larva_ID in larva_IDs:

        df_selected_larva = df_selected.query("larva_ID == @larva_ID").reset_index(level=['larva_ID'], drop=True)[["r", "x", "y"]]

        # Downsample the x,y positions to 1 s
        df_selected_larva.index = pd.to_datetime(df_selected_larva.index, unit='s')  # Convert seconds into datetime objects

        df_selected_larva = df_selected_larva.resample('1s').median()
        df_selected_larva.index = (df_selected_larva.index - pd.to_datetime(0, unit='s')).total_seconds()  # Convert back to seconds
        df_selected_larva.index.rename("time", inplace=True)

        # The speed is defined by dx and dy
        df_selected_larva["speed"] = np.sqrt(df_selected_larva["x"].diff() ** 2 +
                                             df_selected_larva["y"].diff() ** 2) / 1.

        # We only care about everything after 10 as laevae need to get familar with the environment
        for t0, t1 in [[15, 60], [15, 25], [25, 35], [35, 45], [45, 55]]:

            df_selected_larva_time_bin = df_selected_larva.query("time >= @t0*60 and time < @t1*60")

            for r0, r1 in [[0, 2], [2, 4], [4, 6], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]:
                print(experiment_name, larva_ID, t0, t1, r0, r1, len(df_selected_larva_time_bin))

                df_selected_larva_df_selected_larva_time_bin_in_region = df_selected_larva_time_bin.query("r >= @r0 and r < @r1")

                all_results["experiment_name"].append(experiment_name)
                all_results["larva_ID"].append(larva_ID)
                all_results["time_bin"].append(f"t{t0}_to_t{t1}")
                all_results["region_bin"].append(f"r{r0}_to_r{r1}")

                if len(df_selected_larva_time_bin) > 10:
                    all_results["fraction_of_time_spent"].append(100 * len(df_selected_larva_df_selected_larva_time_bin_in_region) / len(df_selected_larva_time_bin))
                else:
                    all_results["fraction_of_time_spent"].append(np.nan)

                all_results["speed"].append(df_selected_larva_df_selected_larva_time_bin_in_region["speed"].median())

df_results = pd.DataFrame.from_dict(all_results)
df_results.set_index(["experiment_name", "larva_ID", "time_bin", "region_bin"], inplace=True)
df_results.sort_index(inplace=True)
print(df_results)


#df_results.to_hdf(root_path / "all_data_model_profile1.h5", key="results_figure1", complevel=9)
df_results.to_hdf(root_path / "all_data_model_profile2.h5", key="results_figure1", complevel=9)

#df_results.to_hdf(root_path / "all_data_deepposekit.h5", key="results_figure1", complevel=9)
#df_results.to_hdf(root_path / "all_data_spatial_phototaxis.h5", key="results_figure1", complevel=9)
