from numba import jit
import random
import pandas as pd
from pathlib import Path
import numpy as np

root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")

@jit(nopython=True)
def luminance_equation_0(x, y):

    r = np.sqrt(x**2 + y**2)

    if r > 5.5:
        return 410*((5.5 - 3) ** 2 / 9 - (r-5.5))
    return 410*((r - 3) ** 2 / 9) ## 1==410 as measured with IPhone

@jit(nopython=True)
def luminance_equation_1(x, y):

    r = np.sqrt(x**2 + y**2)

    if r > 5.5:
        return 410*((5.5 - 3) ** 2 / 9 - (r-5.5))
    return (1 - pow((1 - r / 6), 0.5)) * 410

# l = [luminance_equation_1(x, 0) for x in np.arange(0, 6, 0.1)]
# import pylab as pl
# pl.plot(l)
# pl.show()
# sfd

@jit(nopython=True)
def temporal_photo_taxis_model(experimental_condition, luminance_profile, ts, xs, ys, luminances, orientations, event_indices, rule1, rule2, rule3, rule4, turn_angle_multiplier, run_length_multiplier):

    dt = 0.01
    p_turn_baseline = dt / (15 * run_length_multiplier) # 15 s as measured in the experiment is approx the average
    abs_angle_turn_baseline = 32 * turn_angle_multiplier # 32 deg per turn is approx the average
    speed = 0.04  # cm/s

    previous_event_luminance = 0
    event_counter = 0

    for i in range(1, len(ts)):

        ts[i] = ts[i-1] + dt

        if luminance_profile == 0:
            current_luminance = luminance_equation_0(xs[i - 1], ys[i - 1])
        if luminance_profile == 1:
            current_luminance = luminance_equation_1(xs[i - 1], ys[i - 1])

        p_turn = p_turn_baseline
        abs_angle_turn = abs_angle_turn_baseline

        # if we use the actually luminace, other
        if experimental_condition == True:

            if rule1 == True:
                if current_luminance < 410 * 0.11:  # Dark
                    abs_angle_turn = abs_angle_turn * 0.7
                else:  # Bright
                    abs_angle_turn = abs_angle_turn * 1.3

            if rule2 == True:
                if current_luminance < previous_event_luminance:
                    abs_angle_turn = abs_angle_turn * 0.7
                else:
                    abs_angle_turn = abs_angle_turn * 1.3

            if rule3 == True:
                if current_luminance < 410 * 0.11:  # Dark
                    p_turn = p_turn * 0.7
                else:  # Bright
                    p_turn = p_turn * 1.3

            if rule4 == True:
                if current_luminance < previous_event_luminance:
                    p_turn = p_turn * 0.7
                else:
                    p_turn = p_turn * 1.3

        event_occurred = False
        if random.random() < p_turn:
            orientations[i] = orientations[i - 1] + random.gauss(0, abs_angle_turn)

            event_occurred = True

            previous_event_luminance = current_luminance

        else:
            orientations[i] = orientations[i-1]

        # update the position
        xs[i] = xs[i - 1] + np.cos(np.radians(orientations[i])) * speed * dt
        ys[i] = ys[i - 1] + np.sin(np.radians(orientations[i])) * speed * dt

        if np.sqrt(xs[i]**2 + ys[i]**2) > 6:
            orientations[i] = np.random.random()*360

            xs[i] = xs[i - 1] + np.cos(np.radians(orientations[i])) * speed * dt
            ys[i] = ys[i - 1] + np.sin(np.radians(orientations[i])) * speed * dt

        # always store the experimental luminance, even under control conditions
        luminances[i] = current_luminance

        if event_occurred:
            event_indices[event_counter] = i
            event_counter += 1

    return event_counter

phototaxis_index_grid_search = dict({"rule1": [],
                                     "rule2": [],
                                     "rule3": [],
                                     "rule4": [],
                                     "run_length_multiplier": [],
                                     "turn_angle_multiplier": [],
                                     "phototaxis_index_mean": [],
                                     "phototaxis_index_std": []
                                     })

all_data = dict({"larva_ID": [],
                 "experiment_name": [],
                 "time": [],
                 "x": [],
                 "y": [],
                 "r": [],
                 "roi_movie_framenum": [],
                 "head_x": [],
                 "head_y": [],
                 "center_x": [],
                 "center_y": [],
                 "tail_x": [],
                 "tail_y": []})

all_results = dict({"experiment_name": [],
                    "larva_ID": [],
                    "time_at_current_turn_event": [],
                    "time_since_previous_turn_event": [],
                    "x_at_current_turn_event": [],
                    "y_at_current_turn_event": [],
                    "r_at_current_turn_event": [],
                    "r_at_previous_turn_event": [],
                    "angle_change_at_current_turn_event": [],
                    "luminance_at_current_turn_event": [],
                    "luminance_at_previous_turn_event": [],
                    "luminance_change_since_previous_turn_event": [],
                    "luminance_change_over_last_1s_before_current_turn_event": [],
                    "luminance_change_over_last_2s_before_current_turn_event": [],
                    "luminance_change_over_last_5s_before_current_turn_event": [],
                    "luminance_change_over_last_10s_before_current_turn_event": [],
                    "luminance_change_during_current_turn_event": [],
                    "roi_movie_framenum_at_current_turn_event": [],
                    "curvature_at_current_turn_event": [],
                    "luminance_at_t_minus_20": [],
                    "luminance_at_t_minus_15": [],
                    "luminance_at_t_minus_10": [],
                    "luminance_at_t_minus_5": [],
                    "luminance_at_t_minus_2": [],
                    "luminance_at_t_minus_1": [],
                    "luminance_at_t0": [],
                    "luminance_at_t_plus_1": [],
                    "luminance_at_t_plus_2": [],
                    "luminance_at_t_plus_5": [],
                    "luminance_at_t_plus_10": [],
                    "luminance_at_t_plus_15": [],
                    "luminance_at_t_plus_20": [],
                    })

dt = 0.01
ts = np.zeros(int(60*45/dt))
xs = np.zeros_like(ts)
ys = np.zeros_like(ts)
orientations = np.zeros_like(ts)
luminances = np.zeros_like(ts)

event_indices = np.zeros(10000)


for rule1, rule2, rule3, rule4 in [[False, False, False, False],
                                   [True, False, False, False],
                                   [False, True, False, False],
                                   [False, False, True, False],
                                   [False, False, False, True],
                                   [False, True, False, True],
                                   [True, True, True, True]]:

    for run_length_multiplier in [0.25, 0.5, 1, 2, 4]:
        for turn_angle_multiplier in [0.25, 0.5, 1, 2, 4]:
            print(rule1, rule2, rule3, rule4, run_length_multiplier, turn_angle_multiplier)

            # Do phototaxis index computation
            fraction_of_time_in_dark_ring_experiment = []
            fraction_of_time_in_dark_ring_control = []

            # Experimental
            for larva_i in range(50):
                #print(larva_i)
                while True:
                    xs[0] = np.random.random()*12 - 6
                    ys[0] = np.random.random()*12 - 6
                    if np.sqrt(xs[0]**2 + ys[0]**2) < 6:
                        break

                orientations[0] = np.random.random()*360

                event_counter = temporal_photo_taxis_model(True, 0, ts, xs, ys, luminances, orientations, event_indices, rule1, rule2, rule3, rule4, turn_angle_multiplier, run_length_multiplier)

                # make a summary list for easier computation of the phototaxis index
                r = np.sqrt(xs ** 2 + ys ** 2)

                ind1 = np.where((ts > 15*60) & (ts < 60*60))[0]
                ind2 = np.where((ts > 15*60) & (ts < 60*60) & (r >= 2) & (r < 4))[0]

                fraction_of_time_in_dark_ring_experiment.append(100 * len(ind2) / len(ind1))

            # Control
            for larva_i in range(50):
                #print(larva_i)
                while True:
                    xs[0] = np.random.random()*12 - 6
                    ys[0] = np.random.random()*12 - 6
                    if np.sqrt(xs[0]**2 + ys[0]**2) < 6:
                        break

                event_counter = temporal_photo_taxis_model(False, 0, ts, xs, ys, luminances, orientations, event_indices, rule1, rule2, rule3, rule4, turn_angle_multiplier, run_length_multiplier)

                # make a summary list for easier computation of the phototaxis index
                r = np.sqrt(xs ** 2 + ys ** 2)

                ind1 = np.where((ts > 15*60) & (ts < 60*60))[0]
                ind2 = np.where((ts > 15*60) & (ts < 60*60) & (r >= 2) & (r < 4))[0]

                fraction_of_time_in_dark_ring_control.append(100 * len(ind2) / len(ind1))

            # We define the phototaxis index as the change in time spent in the dark ring relative to control conditions
            ds = []
            for i in range(1000):
                a = np.random.choice(fraction_of_time_in_dark_ring_experiment, 100).mean()
                b = np.random.choice(fraction_of_time_in_dark_ring_control, 100).mean()
                ds.append(a-b)

            phototaxis_index_mean = np.mean(ds)
            phototaxis_index_std = np.std(ds)

            phototaxis_index_grid_search["rule1"].append(rule1)
            phototaxis_index_grid_search["rule2"].append(rule2)
            phototaxis_index_grid_search["rule3"].append(rule3)
            phototaxis_index_grid_search["rule4"].append(rule4)

            phototaxis_index_grid_search["run_length_multiplier"].append(run_length_multiplier)
            phototaxis_index_grid_search["turn_angle_multiplier"].append(turn_angle_multiplier)
            phototaxis_index_grid_search["phototaxis_index_mean"].append(phototaxis_index_mean)
            phototaxis_index_grid_search["phototaxis_index_std"].append(phototaxis_index_std)

df = pd.DataFrame.from_dict(phototaxis_index_grid_search)

df.set_index(["rule1", "rule2", "rule3", "rule4", "run_length_multiplier", "turn_angle_multiplier"], inplace=True)
df.sort_index(inplace=True)

df.to_hdf(root_path / "all_data_model_profile1.h5", key="phototaxis_index_grid_search", complevel=4)

#for experiment_name in ["temporal_phototaxis_drosolarva", "virtual_valley_stimulus_control_drosolarva"]:
for experiment_name in ["virtual_valley_stimulus_drosolarva", "virtual_valley_stimulus_control_drosolarva"]:
    for larva_i in range(50):
        print(experiment_name, larva_i)
        while True:
            xs[0] = np.random.random()*12 - 6
            ys[0] = np.random.random()*12 - 6
            if np.sqrt(xs[0]**2 + ys[0]**2) < 6:
                break

        orientations[0] = np.random.random()*360

        if experiment_name == "virtual_valley_stimulus_drosolarva":
#        if experiment_name == "temporal_phototaxis_drosolarva":
            event_counter = temporal_photo_taxis_model(True, 0, ts, xs, ys, luminances, orientations, event_indices, False, True, False, True, 1, 1)
        if experiment_name == "virtual_valley_stimulus_control_drosolarva":
            event_counter = temporal_photo_taxis_model(False, 0, ts, xs, ys, luminances, orientations, event_indices, False, True, False, True, 1, 1)

        # make a summary list for easier computation of the phototaxis index
        all_data["larva_ID"].extend([larva_i] * len(ts))
        all_data["experiment_name"].extend([experiment_name] * len(ts))

        all_data["time"].extend(ts)
        all_data["x"].extend(xs)
        all_data["y"].extend(ys)
        all_data["r"].extend(np.sqrt(xs ** 2 + ys ** 2) )
        all_data["roi_movie_framenum"].extend([0] * len(ts))
        all_data["head_x"].extend([0] * len(ts))
        all_data["head_y"].extend([0] * len(ts))
        all_data["center_x"].extend([0] * len(ts))
        all_data["center_y"].extend([0] * len(ts))
        all_data["tail_x"].extend([0] * len(ts))
        all_data["tail_y"].extend([0] * len(ts))

        for i in range(1, event_counter):
            current_event_i = int(event_indices[i])
            previous_event_i = int(event_indices[i - 1])

            if current_event_i - int(20/dt) < 0 or current_event_i + int(20/dt) >= len(ts):
                continue

            all_results["experiment_name"].append(experiment_name)
            all_results["larva_ID"].append(f"model_larva_{larva_i}")
            all_results["time_at_current_turn_event"].append(ts[current_event_i])
            all_results["time_since_previous_turn_event"].append(ts[current_event_i] - ts[previous_event_i])
            all_results["x_at_current_turn_event"].append(xs[current_event_i])
            all_results["y_at_current_turn_event"].append(ys[current_event_i])
            all_results["r_at_current_turn_event"].append(np.sqrt(xs[current_event_i]**2 + ys[current_event_i]**2))
            all_results["r_at_previous_turn_event"].append(np.sqrt(xs[previous_event_i] ** 2 + ys[previous_event_i] ** 2))
            all_results["angle_change_at_current_turn_event"].append(orientations[current_event_i + int(1/dt)] -
                                                                     orientations[current_event_i - int(1/dt)])

            all_results["luminance_at_current_turn_event"].append(luminances[current_event_i])
            all_results["luminance_at_previous_turn_event"].append(luminances[previous_event_i])

            all_results["luminance_change_since_previous_turn_event"].append(luminances[current_event_i] - luminances[previous_event_i])
            all_results["luminance_change_over_last_1s_before_current_turn_event"].append(luminances[current_event_i] - luminances[current_event_i - int(1/dt)])
            all_results["luminance_change_over_last_2s_before_current_turn_event"].append(luminances[current_event_i] - luminances[current_event_i - int(2/dt)])
            all_results["luminance_change_over_last_5s_before_current_turn_event"].append(luminances[current_event_i] - luminances[current_event_i - int(5/dt)])
            all_results["luminance_change_over_last_10s_before_current_turn_event"].append(luminances[current_event_i] - luminances[current_event_i - int(10/dt)])
            all_results["luminance_change_during_current_turn_event"].append(luminances[current_event_i + int(1/dt)] - luminances[current_event_i - int(1/dt)])
            all_results["roi_movie_framenum_at_current_turn_event"].append(0)
            all_results["curvature_at_current_turn_event"].append(0)

            all_results["luminance_at_t_minus_20"].append(luminances[current_event_i - int(20/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_minus_15"].append(luminances[current_event_i - int(15/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_minus_10"].append(luminances[current_event_i - int(10/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_minus_5"].append(luminances[current_event_i - int(5/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_minus_2"].append(luminances[current_event_i - int(2/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_minus_1"].append(luminances[current_event_i - int(1/dt)] - luminances[current_event_i])

            all_results["luminance_at_t0"].append(luminances[current_event_i - int(0/dt)] - luminances[current_event_i])

            all_results["luminance_at_t_plus_1"].append(luminances[current_event_i + int(1/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_plus_2"].append(luminances[current_event_i + int(2/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_plus_5"].append(luminances[current_event_i + int(5/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_plus_10"].append(luminances[current_event_i + int(10/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_plus_15"].append(luminances[current_event_i + int(15/dt)] - luminances[current_event_i])
            all_results["luminance_at_t_plus_20"].append(luminances[current_event_i + int(20/dt)] - luminances[current_event_i])


df = pd.DataFrame.from_dict(all_data)

df.set_index(["larva_ID", "experiment_name", "time"], inplace=True)
df.sort_index(inplace=True)

df.to_hdf(root_path / "all_data_model_profile1.h5", key="raw_data")


df_results = pd.DataFrame.from_dict(all_results)
df_results.set_index(["experiment_name", "larva_ID"], inplace=True)
df_results.sort_index(inplace=True)

df_results.to_hdf(root_path / "all_data_model_profile1.h5", key="event_data")
