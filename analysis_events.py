import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib.collections import LineCollection

matplotlib.use("qt5agg")

import pylab as pl

root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")
#df = pd.read_hdf(root_path / "all_data_deepposekit.h5", key="event_data")
df = pd.read_hdf(root_path / "all_data_model_profile1.h5", key="event_data")

print(df)
# Exclude fish
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

#df.to_hdf(root_path / "all_events.h5", key="all_events", complevel=9)
df.to_hdf(root_path / "all_events_model_profile1.h5", key="all_events", complevel=9)


#sdgfsg
#print(df.index.get_level_values(0).unique())
#sdf
#df = pd.read_hdf(root_path / "all_events.h5", key="all_events")
df = pd.read_hdf(root_path / "all_events_model_profile1.h5", key="all_events")


#df[]
df_selected1 = df.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and time_at_current_turn_event > 15*60 and time_at_current_turn_event <= 60*60")
df_selected1.loc[:, "r_at_previous_turn_event"] = df_selected1["r_at_current_turn_event"].shift(1).copy()
df_selected1.loc[:, "r_at_next_turn_event"] = df_selected1["r_at_current_turn_event"].shift(-1).copy()
df_selected1 = df_selected1.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9 and r_at_next_turn_event < 5.9")
#print(df_selected1["r_at_previous_turn_event"], df_selected1["r_at_current_turn_event"])
#sdf
df_selected2 = df.query("experiment_name == 'virtual_valley_stimulus_control_drosolarva' and time_at_current_turn_event > 15*60 and time_at_current_turn_event <= 60*60")
df_selected2.loc[:, "r_at_previous_turn_event"] = df_selected2["r_at_current_turn_event"].shift(1).copy()
df_selected2.loc[:, "r_at_next_turn_event"] = df_selected2["r_at_current_turn_event"].shift(-1).copy()
df_selected2 = df_selected2.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9 and r_at_next_turn_event < 5.9")

from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()  # create object for the class


#X = df_selected1["luminance_at_current_turn_event"].values.reshape(-1, 1)
#Y = df_selected1["angle_change_at_current_turn_event"].abs().values.reshape(-1, 1)
#reg = linear_regressor.fit(X, Y)  # perform linear regression
#Y_pred = linear_regressor.predict(X)  # make predictions

#print(reg.score(X, Y))
#
# pl.plot(df_selected1["luminance_at_current_turn_event"], df_selected1["angle_change_at_current_turn_event"].abs(), '.', alpha=0.1)
# #pl.plot(X, Y_pred)
#
# vals = []
# bins = np.arange(0, 300, 40)
# for bin in bins:
#
#     df_ = df_selected1.query("luminance_at_current_turn_event < (@bin + 20) and luminance_at_current_turn_event > (@bin - 20)")
#     vals.append(df_["angle_change_at_current_turn_event"].abs().median())
# pl.plot(bins, vals, 'ro')
# X = np.array(bins).reshape(-1, 1)
# Y = np.array(vals).reshape(-1, 1)
#
# print(X,Y)
# reg = linear_regressor.fit(X, Y)  # perform linear regression
#
# Y_pred = linear_regressor.predict(X)  # make predictions
# pl.plot(bins, Y_pred, 'black')
# print(reg.score(X, Y))
#
# pl.figure()
# pl.plot(df_selected2["luminance_change_since_previous_turn_event"], df_selected2["angle_change_at_current_turn_event"].abs(), '.', alpha=0.1)
# #pl.plot(X, Y_pred)
#
# for bin in np.arange(-100, 100, 20):
#     df_ = df_selected2.query("luminance_change_since_previous_turn_event < (@bin + 10) and luminance_change_since_previous_turn_event > (@bin - 10)")
#     pl.plot(bin, df_["angle_change_at_current_turn_event"].abs().median(), 'ro')
#
#
# pl.show()
# dfd
#
#
# pl.show()
# dfg

#
# and r_at_current_turn_event < 5.6 and r_at_previous_turn_event < 5.6")
# df_selected2 = df.query("experiment_name == 'virtual_valley_stimulus_control_drosolarva' and time_at_current_turn_event > 15*60 and time_at_current_turn_event <= 60*60 and r_at_current_turn_event < 5.6 and r_at_previous_turn_event < 5.6")

#print(df_selected[["luminance_at_t_minus_5", "luminance_at_t_plus_5"]])
k1 = df_selected1[["luminance_at_t_minus_20",
                 "luminance_at_t_minus_15",
                "luminance_at_t_minus_10",
                "luminance_at_t_minus_5",
                "luminance_at_t_minus_2",
                "luminance_at_t_minus_1",
                 "luminance_at_t0",
                "luminance_at_t_plus_1",
                "luminance_at_t_plus_2",
                "luminance_at_t_plus_5",
                "luminance_at_t_plus_10",
                "luminance_at_t_plus_15",
                 "luminance_at_t_plus_20"]]

#print(df_selected[["luminance_at_t_minus_5", "luminance_at_t_plus_5"]])
k2 = df_selected2[["luminance_at_t_minus_20",
                   "luminance_at_t_minus_15",
                   "luminance_at_t_minus_10",
                   "luminance_at_t_minus_5",
                   "luminance_at_t_minus_2",
                   "luminance_at_t_minus_1",
                   "luminance_at_t0",
                   "luminance_at_t_plus_1",
                   "luminance_at_t_plus_2",
                   "luminance_at_t_plus_5",
                   "luminance_at_t_plus_10",
                   "luminance_at_t_plus_15",
                   "luminance_at_t_plus_20"]]

#print(k1.mean(axis=0))
#sdf
print(len(k1), len(k2))
sdf
all_results = dict({"bins": [-20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20],
                    "means_experiment": k1.mean(axis=0),
                    "sems_experiment": k1.sem(axis=0),
                    "means_control": k2.mean(axis=0),
                    "sems_control": k2.sem(axis=0)})


df_results = pd.DataFrame.from_dict(all_results)
df_results.set_index(["bins"], inplace=True)
df_results.sort_index(inplace=True)
df_results.to_hdf(root_path / "all_events_model_profile1.h5", key="event_triggered_luminance", complevel=9)
#df_results.to_hdf(root_path / "all_events.h5", key="event_triggered_luminance", complevel=9)

#pl.errorbar([-20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20], k1.mean(axis=0), k1.sem(axis=0), color='C0')

#pl.plot([-20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20], k2.mean(axis=0), '-o', color='C1')
#pl.errorbar([-20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20], k2.mean(axis=0), k2.sem(axis=0), color='C1')

#pl.plot([-20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20], k.quantile(q=0.75, axis=0))
#pl.plot([-20, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 20], k.quantile(q=0.11, axis=0))

#pl.show()
#sdf
#print(df)
#sdf
all_results = dict({"experiment_name": [],
                    "larva_ID": [],

                    "angle_change_at_current_turn_event_if_bright_at_current_turn_event": [],
                    "angle_change_at_current_turn_event_if_dark_at_current_turn_event": [],
                    #"angle_change_at_current_turn_event_if_bright_at_previous_turn_event": [],
                    #"angle_change_at_current_turn_event_if_dark_at_previous_turn_event": [],

                    "angle_change_at_current_turn_event_if_brightening_since_previous_turn_event": [],
                    #"angle_change_at_current_turn_event_if_medium_brightening_since_previous_turn_event": [],
                    #"angle_change_at_current_turn_event_if_strong_brightening_since_previous_turn_event": [],

                    #"angle_change_at_current_turn_event_if_brightening_over_last_10s_before_current_turn_event": [],
                    #"angle_change_at_current_turn_event_if_brightening_over_last_5s_before_current_turn_event": [],
                    #"angle_change_at_current_turn_event_if_brightening_over_last_2s_before_current_turn_event": [],
                    #"angle_change_at_current_turn_event_if_brightening_over_last_1s_before_current_turn_event": [],

                    "angle_change_at_current_turn_event_if_darkening_since_previous_turn_event": [],
                    #"angle_change_at_current_turn_event_if_medium_darkening_since_previous_turn_event": [],
                    #"angle_change_at_current_turn_event_if_strong_darkening_since_previous_turn_event": [],

                    #"angle_change_at_current_turn_event_if_darkening_over_last_10s_before_current_turn_event": [],
                    #"angle_change_at_current_turn_event_if_darkening_over_last_5s_before_current_turn_event": [],
                    #"angle_change_at_current_turn_event_if_darkening_over_last_2s_before_current_turn_event": [],
                    #"angle_change_at_current_turn_event_if_darkening_over_last_1s_before_current_turn_event": [],

                    "time_since_previous_turn_event_at_current_turn_event_if_bright_at_current_turn_event": [],
                    "time_since_previous_turn_event_at_current_turn_event_if_dark_at_current_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_bright_at_previous_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_dark_at_previous_turn_event": [],

                    "time_since_previous_turn_event_at_current_turn_event_if_brightening_since_previous_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_medium_brightening_since_previous_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_strong_brightening_since_previous_turn_event": [],

                    #"time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_10s_before_current_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_5s_before_current_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_2s_before_current_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_1s_before_current_turn_event": [],

                    "time_since_previous_turn_event_at_current_turn_event_if_darkening_since_previous_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_medium_darkening_since_previous_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_strong_darkening_since_previous_turn_event": [],

                    #"time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_10s_before_current_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_5s_before_current_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_2s_before_current_turn_event": [],
                    #"time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_1s_before_current_turn_event": [],

                    "luminance_change_since_previous_turn_event": [],
                    #"luminance_change_over_last_10s_before_current_turn_event": [],
                    #"luminance_change_over_last_5s_before_current_turn_event": [],
                    #"luminance_change_over_last_2s_before_current_turn_event": [],
                    #"luminance_change_over_last_1s_before_current_turn_event": [],
                    "luminance_change_during_current_turn_event": []
                })

histogram_results = dict({"experiment_name": [],
                          "histogram_type": [],
                          "bin": [],
                          "density": []})

experiment_names = df.index.get_level_values('experiment_name').unique().values
print(experiment_names)
for experiment_name in experiment_names:

    df_selected_experiment = df.query("experiment_name == @experiment_name and time_at_current_turn_event > 15*60 and time_at_current_turn_event <= 60*60").reset_index(level=['experiment_name'], drop=True)
    larva_IDs = df_selected_experiment.index.get_level_values('larva_ID').unique().values
    for larva_ID in larva_IDs:

        df_selected_larva = df_selected_experiment.query("larva_ID == @larva_ID").reset_index(level=['larva_ID'], drop=True)

        print(larva_ID, df_selected_larva["time_since_previous_turn_event"].abs().median(),
              df_selected_larva["angle_change_at_current_turn_event"].abs().median())
        all_results["experiment_name"].append(experiment_name)
        all_results["larva_ID"].append(larva_ID)

        all_results["angle_change_at_current_turn_event_if_bright_at_current_turn_event"].append(df_selected_larva.query("luminance_at_current_turn_event > 0.11*410 and r_at_current_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        all_results["angle_change_at_current_turn_event_if_dark_at_current_turn_event"].append(df_selected_larva.query("luminance_at_current_turn_event <= 0.11*410 and r_at_current_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())

        #all_results["angle_change_at_current_turn_event_if_bright_at_previous_turn_event"].append(df_selected_larva.query("luminance_at_previous_turn_event > 0.11 and r_at_current_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_dark_at_previous_turn_event"].append(df_selected_larva.query("luminance_at_previous_turn_event <= 0.11 and r_at_current_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())

        all_results["angle_change_at_current_turn_event_if_brightening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_medium_brightening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event > 0 and luminance_change_over_last_2s_before_current_turn_event < 0.025 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_strong_brightening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event > 0.025 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #
        # all_results["angle_change_at_current_turn_event_if_brightening_over_last_10s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_10s_before_current_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        # all_results["angle_change_at_current_turn_event_if_brightening_over_last_5s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_5s_before_current_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        # all_results["angle_change_at_current_turn_event_if_brightening_over_last_2s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        # all_results["angle_change_at_current_turn_event_if_brightening_over_last_1s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_1s_before_current_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())

        #all_results["angle_change_at_current_turn_event_if_brightening_over_last_10s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event > 0 and time_since_previous_turn_event < 7.5 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_brightening_over_last_5s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event > 0 and time_since_previous_turn_event < 15 and time_since_previous_turn_event >= 7.5 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_brightening_over_last_2s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event > 0 and time_since_previous_turn_event < 22.5 and time_since_previous_turn_event >= 15 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_brightening_over_last_1s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event > 0 and time_since_previous_turn_event > 22.5 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())


        all_results["angle_change_at_current_turn_event_if_darkening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_medium_darkening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event > -0.2 and luminance_change_over_last_5s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_strong_darkening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event < -0.2 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())

        # all_results["angle_change_at_current_turn_event_if_darkening_over_last_10s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_10s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        # all_results["angle_change_at_current_turn_event_if_darkening_over_last_5s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_5s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        # all_results["angle_change_at_current_turn_event_if_darkening_over_last_2s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        # all_results["angle_change_at_current_turn_event_if_darkening_over_last_1s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_1s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())

        #all_results["angle_change_at_current_turn_event_if_darkening_over_last_10s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event < 0 and time_since_previous_turn_event < 7.5 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_darkening_over_last_5s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event < 0 and time_since_previous_turn_event < 15 and time_since_previous_turn_event >= 7.5 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_darkening_over_last_2s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event < 0 and time_since_previous_turn_event < 22.5 and time_since_previous_turn_event >= 15 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())
        #all_results["angle_change_at_current_turn_event_if_darkening_over_last_1s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event < 0 and time_since_previous_turn_event > 22.5 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].abs().median())


        all_results["time_since_previous_turn_event_at_current_turn_event_if_bright_at_current_turn_event"].append(df_selected_larva.query("luminance_at_current_turn_event > 0.11*410 and r_at_current_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        all_results["time_since_previous_turn_event_at_current_turn_event_if_dark_at_current_turn_event"].append(df_selected_larva.query("luminance_at_current_turn_event <= 0.11*410 and r_at_current_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())

        #all_results["time_since_previous_turn_event_at_current_turn_event_if_bright_at_previous_turn_event"].append(df_selected_larva.query("luminance_at_previous_turn_event >0.11 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_dark_at_previous_turn_event"].append(df_selected_larva.query("luminance_at_previous_turn_event <=0.11 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())

        all_results["time_since_previous_turn_event_at_current_turn_event_if_brightening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_medium_brightening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event > 0 and luminance_change_over_last_2s_before_current_turn_event < 0.025 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_strong_brightening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event > 0.025 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())

        #all_results["time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_10s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_10s_before_current_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_5s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_5s_before_current_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_2s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_1s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_1s_before_current_turn_event > 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())

        all_results["time_since_previous_turn_event_at_current_turn_event_if_darkening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_since_previous_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_medium_darkening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event > -0.025 and luminance_change_over_last_2s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_strong_darkening_since_previous_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event < -0.025 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())

        #all_results["time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_10s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_10s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_5s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_5s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_2s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_2s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())
        #all_results["time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_1s_before_current_turn_event"].append(df_selected_larva.query("luminance_change_over_last_1s_before_current_turn_event < 0 and r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].abs().median())

        all_results["luminance_change_since_previous_turn_event"].append(df_selected_larva.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["luminance_change_since_previous_turn_event"].abs().median())
        #all_results["luminance_change_over_last_10s_before_current_turn_event"].append(df_selected_larva.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["luminance_change_over_last_10s_before_current_turn_event"].abs().median())
        #all_results["luminance_change_over_last_5s_before_current_turn_event"].append(df_selected_larva.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["luminance_change_over_last_5s_before_current_turn_event"].abs().median())
        #all_results["luminance_change_over_last_2s_before_current_turn_event"].append(df_selected_larva.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["luminance_change_over_last_2s_before_current_turn_event"].abs().median())
        #all_results["luminance_change_over_last_1s_before_current_turn_event"].append(df_selected_larva.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["luminance_change_over_last_1s_before_current_turn_event"].abs().median())
        all_results["luminance_change_during_current_turn_event"].append(df_selected_larva.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["luminance_change_during_current_turn_event"].abs().median())

    bins = np.linspace(-180, 180, 60)
    density, _ = np.histogram(df_selected_experiment.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["angle_change_at_current_turn_event"].values, bins=bins, density=True)

    histogram_results["experiment_name"].extend([experiment_name] * (len(bins) - 1))
    histogram_results["histogram_type"].extend(["angle_change"] * (len(bins) - 1))
    histogram_results["bin"].extend(bins[1:] - np.diff(bins)/2)
    histogram_results["density"].extend(density)

    ####
    bins = np.linspace(0, 60, 60)
    density, _ = np.histogram(df_selected_experiment.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["time_since_previous_turn_event"].values, bins=bins, density=True)

    histogram_results["experiment_name"].extend([experiment_name] * (len(bins) - 1))
    histogram_results["histogram_type"].extend(["run_length"] * (len(bins) - 1))
    histogram_results["bin"].extend(bins[1:] - np.diff(bins)/2)
    histogram_results["density"].extend(density)

    ####
    bins = np.linspace(-80, 80, 60)
    density, _ = np.histogram(df_selected_experiment.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["luminance_change_since_previous_turn_event"].values, bins=bins, density=True)

    histogram_results["experiment_name"].extend([experiment_name] * (len(bins) - 1))
    histogram_results["histogram_type"].extend(["luminance_change_since_previous_turn_event"] * (len(bins) - 1))
    histogram_results["bin"].extend(bins[1:] - np.diff(bins)/2)
    histogram_results["density"].extend(density)


    bins = np.linspace(-80, 80, 60)
    density, _ = np.histogram(df_selected_experiment.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9")["luminance_change_during_current_turn_event"].values, bins=bins, density=True)

    histogram_results["experiment_name"].extend([experiment_name] * (len(bins) - 1))
    histogram_results["histogram_type"].extend(["luminance_change_during_current_turn_event"] * (len(bins) - 1))
    histogram_results["bin"].extend(bins[1:] - np.diff(bins)/2)
    histogram_results["density"].extend(density)



df_results = pd.DataFrame.from_dict(all_results)
df_results.set_index(["experiment_name", "larva_ID"], inplace=True)
df_results.sort_index(inplace=True)

df_histogram_results = pd.DataFrame.from_dict(histogram_results)
df_histogram_results.set_index(["experiment_name", "histogram_type", "bin"], inplace=True)
df_histogram_results.sort_index(inplace=True)

df_results.to_hdf(root_path / "all_events.h5", key="results_figure2", complevel=9)
df_histogram_results.to_hdf(root_path / "all_events.h5", key="results_figure2_histograms", complevel=9)

#df_results.to_hdf(root_path / "all_events_model_profile1.h5", key="results_figure2", complevel=9)
#df_histogram_results.to_hdf(root_path / "all_events_model_profile1.h5", key="results_figure2_histograms", complevel=9)



print("Done")
