import pandas as pd
from pathlib import Path
import numpy as np
import pylab as pl
from scipy.signal import find_peaks
from my_general_helpers import butter_lowpass_filter

def angle_between_points_signcorrect(x1, y1, x2, y2, x3, y3):
    ang1 = np.degrees(np.arctan2(y1 - y2, x1 - x2))
    ang2 = np.degrees(np.arctan2(y3 - y2, x3 - x2))

    if np.ndim(x1) == 0:
        if ang1 < 0:
            ang1 = 360 + ang1
        if ang2 < 0:
            ang2 = 360 + ang2

        if ang2 > ang1:
            ang2 -= 360
    else:
        ind = np.where(ang1 < 0)
        ang1[ind] = 360 + ang1[ind]

        ind = np.where(ang2 < 0)
        ang2[ind] = 360 + ang2[ind]

        ind = np.where(ang2 > ang1)
        ang2[ind] -= 360

    return (ang1 - ang2) - 180


def curvature(x1, y1, x2, y2, x3, y3):#, x4, y4, x5, y5):
    dx1 = x1 - x2
    dy1 = y1 - y2
    dx2 = x2 - x3
    dy2 = y2 - y3

    # dx3 = x2 - x3
    # dy3 = y2 - y3
    # dx4 = x3 - x4
    # dy4 = y3 - y4
    #
    # dx5 = x3 - x4
    # dy5 = y3 - y4
    # dx6 = x4 - x5
    # dy6 = y4 - y5

    dotProduct1 = dx1 * dx2 + dy1 * dy2
    modOfVectors1 = np.sqrt(dx1**2 + dy1**2) * np.sqrt(dx2**2 + dy2**2)
    #
    # dotProduct2 = dx3 * dx4 + dy3 * dy4
    # modOfVectors2 = np.sqrt(dx3**2 + dy3**2) * np.sqrt(dx4**2 + dy4**2)
    #
    # dotProduct3 = dx5 * dx6 + dy5 * dy6
    # modOfVectors3 = np.sqrt(dx5**2 + dy5**2) * np.sqrt(dx6**2 + dy6**2)

    return np.degrees(np.arccos(dotProduct1/modOfVectors1))# + \
           #np.degrees(np.arccos(dotProduct2/modOfVectors2)) + \
           #np.degrees(np.arccos(dotProduct3/modOfVectors3))


def luminance_equation(x, y):

    r = np.sqrt(x**2 + y**2)

    if r > 5.5:
        return 410*((5.5 - 3) ** 2 / 9 - (r-5.5))
    return 410*((r - 3) ** 2 / 9) ## 1==410 as measured with IPhone

print(luminance_equation(3.9, 0))
print(luminance_equation(4, 0))
print(luminance_equation(4.1, 0))

#sdf
# vals = [luminance_equation(x, 0) for x in np.arange(0, 6, 0.1)]
# pl.plot(vals)
# pl.show()
# dfg
root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")
df = pd.read_hdf(root_path / "all_data_deepposekit.h5", key="raw_data")

df["curvature"] = curvature(df["head_x"].values,
                            df["head_y"].values,
                            #df["head_node_x"].values,
                            #df["head_node_y"].values,
                            df["center_x"].values,
                            df["center_y"].values,
                            #df["tail_node_x"].values,
                            #df["tail_node_y"].values,
                            df["tail_x"].values,
                            df["tail_y"].values)

df["curvature"] = df["curvature"].fillna(method='bfill')

# Filter out all jumps and lost larvae
df["curvature_filtered"] = butter_lowpass_filter(df["curvature"], cutoff=3, fs=90., order=5)
# pl.plot(df["curvature"].values)
# pl.plot(df["curvature_filtered"].values)
# pl.show()
# print(df["curvature_filtered"])

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

experiment_names = df.index.get_level_values('experiment_name').unique().values

for experiment_name in experiment_names:

    df_selected = df.query("experiment_name == @experiment_name").reset_index(level=['experiment_name'], drop=True)
    larva_IDs = df_selected.index.get_level_values('larva_ID').unique().values
    for larva_ID in larva_IDs:
        #if larva_ID != "2018_11_15_fish006_setup1":
        #    continue

        # print(experiment_name, larva_ID)
        df_selected_larva = df_selected.query("larva_ID == @larva_ID").reset_index(level=['larva_ID'], drop=True)

        # pl.plot(df_selected_larva["x"], df_selected_larva["y"])
        #
        # pl.plot(df_selected_larva["x"] + (df_selected_larva["center_x"] - 50)*0.0002,
        #         df_selected_larva["y"] + (df_selected_larva["center_y"] - 50)*0.0002)
        #
        # pl.show()
        # pl.plot(df_selected_larva['curvature'])
        # pl.plot(df_selected_larva['curvature_filtered'])
        #
        # # Downsample the x,y positions to 1 s
        # df_selected_larva.index = pd.to_datetime(df_selected_larva.index, unit='s')  # Convert seconds into datetime objects
        #
        # df_selected_larva = df_selected_larva.resample('3s').median()
        # df_selected_larva.index = (df_selected_larva.index - pd.to_datetime(0, unit='s')).total_seconds()  # Convert back to seconds
        # df_selected_larva.index.rename("time", inplace=True)
        #
        # # The speed is defined by dx and dy
        # df_selected_larva["speed"] = np.sqrt(df_selected_larva["x"].diff() ** 2 +
        #                                      df_selected_larva["y"].diff() ** 2) / 1.
        #
        # pl.plot(df_selected_larva["speed"]*100)
        # pl.show()
        # fff
        #
        # # Show the sliding window variance of curvature

        #sdf
        
        peaks, _ = find_peaks(df_selected_larva['curvature_filtered'], height=30, distance=2 * 90, width=0.5 * 90, prominence=0.5)
        #
        # if larva_ID == "2018_11_15_fish007_setup1":
        #     #pl.plot(df_selected_larva['curvature'])
        #     #pl.figure()
        #     pl.plot(df_selected_larva['curvature_filtered'])
        #     pl.plot(df_selected_larva['curvature_filtered'].iloc[peaks], 'o')
        #     pl.show()
        #     sdf

        # # Clean up peaks (spatial distance and height)
        # for i in range(1, len(peaks)):
        #     if np.sqrt((df_selected_larva.iloc[peaks[i]]["x"] - df_selected_larva.iloc[peaks[i - 1]]["x"]) ** 2 +
        #                (df_selected_larva.iloc[peaks[i]]["y"] - df_selected_larva.iloc[peaks[i - 1]]["y"]) ** 2) < 0.3: # should be at least 3mm apart
        #         peaks[i] = -1 # Remove this peak
        #
        #     if df_selected_larva['curvature_filtered'].iloc[peaks[i]] > 160:
        #         peaks[i] = -1

        #peaks = peaks[np.where(peaks > -1)]

        previous_x = np.nan
        previous_y = np.nan
        previous_time = np.nan
        previous_r = np.nan
        previous_luminance = np.nan

        for peak_i in peaks:
            #print(peak_i)
            #continue
            if peak_i - 20*90 < 0:
                continue
            if peak_i + 20*90 >= len(df_selected_larva):
                continue

            current_x = df_selected_larva.iloc[peak_i]["x"]
            current_y = df_selected_larva.iloc[peak_i]["y"]
            current_r = np.sqrt(current_x**2 + current_y**2)
            current_time = df_selected_larva.iloc[peak_i].name
            current_luminance = luminance_equation(current_x, current_y)

            current_angle_change = angle_between_points_signcorrect(df_selected_larva.iloc[peak_i - 180]["x"],
                                                                    df_selected_larva.iloc[peak_i - 180]["y"],
                                                                    df_selected_larva.iloc[peak_i]["x"],
                                                                    df_selected_larva.iloc[peak_i]["y"],
                                                                    df_selected_larva.iloc[peak_i + 180]["x"],
                                                                    df_selected_larva.iloc[peak_i + 180]["y"])

            # Probably a mistake in the detection of the position
            #if np.abs(current_angle_change) > 150:
            #    continue

            luminance_change_during_turn_event = luminance_equation(df_selected_larva.iloc[peak_i + 90]["x"], df_selected_larva.iloc[peak_i + 90]["y"]) - \
                                                 luminance_equation(df_selected_larva.iloc[peak_i - 90]["x"], df_selected_larva.iloc[peak_i - 90]["y"])

            # Ignore if too close in spatial distance to previous event
            if np.isnan(previous_x) or np.sqrt((current_x - previous_x)**2 +
                                               (current_y - previous_y)**2) > 0.2:

                all_results["experiment_name"].append(experiment_name)
                all_results["larva_ID"].append(larva_ID)
                all_results["time_at_current_turn_event"].append(current_time)
                all_results["time_since_previous_turn_event"].append(current_time - previous_time)

                all_results["x_at_current_turn_event"].append(current_x)
                all_results["y_at_current_turn_event"].append(current_y)
                all_results["r_at_current_turn_event"].append(current_r)
                all_results["r_at_previous_turn_event"].append(previous_r)
                all_results["luminance_at_current_turn_event"].append(current_luminance)
                all_results["luminance_at_previous_turn_event"].append(previous_luminance)

                all_results["angle_change_at_current_turn_event"].append(current_angle_change)
                all_results["luminance_change_since_previous_turn_event"].append(current_luminance - previous_luminance)
                all_results["luminance_change_over_last_1s_before_current_turn_event"].append(current_luminance - luminance_equation(df_selected_larva.iloc[peak_i-90]["x"], df_selected_larva.iloc[peak_i-90]["y"]))
                all_results["luminance_change_over_last_2s_before_current_turn_event"].append(current_luminance - luminance_equation(df_selected_larva.iloc[peak_i-180]["x"], df_selected_larva.iloc[peak_i-180]["y"]))
                all_results["luminance_change_over_last_5s_before_current_turn_event"].append(current_luminance - luminance_equation(df_selected_larva.iloc[peak_i-450]["x"], df_selected_larva.iloc[peak_i-450]["y"]))
                all_results["luminance_change_over_last_10s_before_current_turn_event"].append(current_luminance - luminance_equation(df_selected_larva.iloc[peak_i-900]["x"], df_selected_larva.iloc[peak_i-900]["y"]))
                all_results["luminance_change_during_current_turn_event"].append(luminance_change_during_turn_event)

                all_results["roi_movie_framenum_at_current_turn_event"].append(df_selected_larva.iloc[peak_i]["roi_movie_framenum"])
                all_results["curvature_at_current_turn_event"].append(df_selected_larva.iloc[peak_i]["curvature"])


                all_results["luminance_at_t_minus_20"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i - 20 * 90]["x"],
                                       df_selected_larva.iloc[peak_i - 20 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_minus_15"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i - 15 * 90]["x"],
                                       df_selected_larva.iloc[peak_i - 15 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_minus_10"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i - 10 * 90]["x"],
                                       df_selected_larva.iloc[peak_i - 10 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_minus_5"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i - 5 * 90]["x"],
                                       df_selected_larva.iloc[peak_i - 5 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_minus_2"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i - 2 * 90]["x"],
                                       df_selected_larva.iloc[peak_i - 2 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_minus_1"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i - 1 * 90]["x"],
                                       df_selected_larva.iloc[peak_i - 1 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t0"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i]["x"],
                                       df_selected_larva.iloc[peak_i]["y"]) - current_luminance)

                all_results["luminance_at_t_plus_1"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i + 1 * 90]["x"],
                                       df_selected_larva.iloc[peak_i + 1 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_plus_2"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i + 2 * 90]["x"],
                                       df_selected_larva.iloc[peak_i + 2 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_plus_5"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i + 5 * 90]["x"],
                                       df_selected_larva.iloc[peak_i + 5 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_plus_10"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i + 10 * 90]["x"],
                                       df_selected_larva.iloc[peak_i + 10 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_plus_15"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i + 15 * 90]["x"],
                                       df_selected_larva.iloc[peak_i + 15 * 90]["y"]) - current_luminance)

                all_results["luminance_at_t_plus_20"].append(
                    luminance_equation(df_selected_larva.iloc[peak_i + 20 * 90]["x"],
                                       df_selected_larva.iloc[peak_i + 20 * 90]["y"]) - current_luminance)

                previous_x = current_x
                previous_y = current_y
                previous_r = current_r
                previous_time = current_time
                previous_luminance = current_luminance

df_results = pd.DataFrame.from_dict(all_results)
df_results.set_index(["experiment_name", "larva_ID"], inplace=True)
df_results.sort_index(inplace=True)

df_results.to_hdf(root_path / "all_data_deepposekit.h5", key="event_data", complevel=9)
