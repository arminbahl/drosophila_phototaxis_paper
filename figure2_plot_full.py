import pandas as pd
from pathlib import Path
import pylab as pl
import my_figure as myfig
from scipy.stats import ttest_ind, ttest_1samp
import numpy as np


root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")

df = pd.read_hdf(root_path / "all_events.h5", key="results_figure2")
df_histogram_results = pd.read_hdf(root_path / "all_events.h5", key="results_figure2_histograms")
df_event_triggered_luminance = pd.read_hdf(root_path / "all_events.h5", key="event_triggered_luminance")

df.to_excel(root_path / "all_events_figure2.xlsx", sheet_name="all_events.h5")
df.groupby("experiment_name").mean().to_excel(root_path / "all_events_figure2_experiment_mean.xlsx", sheet_name="all_data")


fig = myfig.Figure(title="Figure 2")

##########
p0 = myfig.Plot(fig, num='a', xpos=1.5, ypos=22, plot_height=1.25, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Turn angle (deg)", xmin=-181, xmax=181, xticks=[-180, -90, 0, 90, 180],
                    yl="Probability density", ymin=-0.001, ymax=0.01)

df_selected = df_histogram_results.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and histogram_type == 'angle_change'").reset_index(level=['experiment_name', 'histogram_type'], drop=True)
myfig.Bar(p0, x=df_selected.index, y=df_selected["density"].values, lc='C0', lw=0, width=0.95*362/60)

##########
p0 = myfig.Plot(fig, num='b', xpos=4.5, ypos=22, plot_height=1.25, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Run length (s)", xmin=-0.5, xmax=60.5, xticks=[0, 30, 60],
                    yl="Probability density", ymin=-0.001, ymax=0.06)

df_selected = df_histogram_results.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and histogram_type == 'run_length'").reset_index(level=['experiment_name', 'histogram_type'], drop=True)
myfig.Bar(p0, x=df_selected.index, y=df_selected["density"].values, lc='C0', lw=0, width=0.95*61/60)

##########
for experiment_name, x_pos, y_pos, color in [["virtual_valley_stimulus_drosolarva", 19, 19, "C0"],
                                             ["virtual_valley_stimulus_control_drosolarva", 19, 15, "gray"]]:
    p0 = myfig.Plot(fig, num='e', xpos=x_pos, ypos=y_pos, plot_height=1.25, plot_width=0.375*5,
                        lw=1, pc='white', errorbar_area=False,
                        xl="", xmin=-0.5, xmax=4.5, xticks=[0, 1, 2, 3, 4], xticklabels=["Since previous turn event",
                                                                                         "Over last 10 s before current turn event",
                                                                                         "Over last 5 s before current turn event",
                                                                                         "Over last 2 s before current turn event",
                                                                                         "Over last 1 s before current turn event"], xticklabels_rotation=45,
                        yl="Luminance change", ymin=-0.01, ymax=0.21, yticks=[0, 0.1, 0.2])


    for j in range(len(df.query("experiment_name == @experiment_name"))):

        for i in range(5):
            if i == 0:
                luminance_change = df.query("experiment_name == @experiment_name")[f"luminance_change_since_previous_turn_event"]
            if i == 1:
                luminance_change = df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_10s_before_current_turn_event"]
            if i == 2:
                luminance_change = df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_5s_before_current_turn_event"]
            if i == 3:
                luminance_change = df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_2s_before_current_turn_event"]
            if i == 4:
                luminance_change = df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_1s_before_current_turn_event"]

            x = np.random.random() * 0.2 - 0.1 + i
            y = luminance_change[j]

            myfig.Scatter(p0, x=[x], y=[y], lc=color, pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)

            if i > 0:
                myfig.Line(p0, x=[x, previous_x], y=[y, previous_y], lc=color, lw=0.5, zorder=1, alpha=0.5)

            previous_x = x
            previous_y = y

    for i in range(4):
        if i == 0:
            p = ttest_1samp(df.query("experiment_name == @experiment_name")[f"luminance_change_since_previous_turn_event"] -
                            df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_10s_before_current_turn_event"], 0, nan_policy='omit')[1]
        if i == 1:
            p = ttest_1samp(df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_10s_before_current_turn_event"] -
                            df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_5s_before_current_turn_event"], 0, nan_policy='omit')[1]
        if i == 2:
            p = ttest_1samp(df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_5s_before_current_turn_event"] -
                            df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_2s_before_current_turn_event"], 0, nan_policy='omit')[1]
        if i == 3:
            p = ttest_1samp(df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_2s_before_current_turn_event"] -
                            df.query("experiment_name == @experiment_name")[f"luminance_change_over_last_1s_before_current_turn_event"], 0, nan_policy='omit')[1]

        myfig.Line(p0, x=[0.1 + i, 0.9 + i], y=[0.15, 0.15], lc='black', lw=0.75)
        if p < 0.001:
            myfig.Text(p0, 0.5 + i, 0.17, "***")
        elif p < 0.01:
            myfig.Text(p0, 0.5 + i, 0.17, "**")
        elif p < 0.05:
            myfig.Text(p0, 0.5 + i, 0.17, "*")
        else:
            myfig.Text(p0, 0.5 + i, 0.17, "ns")

##########
for experiment_name, x_pos, y_pos, color in [["virtual_valley_stimulus_drosolarva", 2, 19, "C0"],
                                             ["virtual_valley_stimulus_control_drosolarva", 2, 17, "gray"]]:

    p0 = myfig.Plot(fig, num='b', xpos=x_pos, ypos=y_pos, plot_height=1.25, plot_width=0.375*24,
                    lw=1, pc='white', errorbar_area=False,
                    xl="", xmin=-0.5, xmax=23.5, xticks=[0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23],
                    xticklabels=["Dark at current turn event",
                                 "Bright at current turn event",
                                 "Dark at previous turn event",
                                 "Bright at previous turn event",

                                 "Darkening since previous turn event",
                                 "Brightening since previous turn event",
                                 "Darkening over last 10 s before current turn event",
                                 "Brightening over last 10 s before current turn event",
                                 "Darkening over last 5 s before current turn event",
                                 "Brightening over last 5 s before current turn event",
                                 "Darkening over last 2 s before current turn event",
                                 "Brightening over last 2 s before current turn event",
                                 "Darkening over last 1 s before current turn event",
                                 "Brightening over last 1 s before current turn event",

                                 "Medium darkening since last turn event",
                                 "Strong darkening since last turn event",
                                 "Medium brightening since last turn event",
                                 "Strong brightening since last turn event"] if experiment_name == "virtual_valley_stimulus_control_drosolarva" else [""]*18,
                    xticklabels_rotation=45,
                    yl="Absolute angle change (°)", ymin=-5, ymax=65, yticks=[0, 30, 60])

    for i in range(9):
        if i == 0:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_dark_at_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_bright_at_current_turn_event"]
            #print(angle_change1)
            #d#fg
        if i == 1:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_dark_at_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_bright_at_previous_turn_event"]

        if i == 2:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_darkening_since_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_brightening_since_previous_turn_event"]

        if i == 3:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_darkening_over_last_10s_before_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_brightening_over_last_10s_before_current_turn_event"]

        if i == 4:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_darkening_over_last_5s_before_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_brightening_over_last_5s_before_current_turn_event"]

        if i == 5:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_darkening_over_last_2s_before_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_brightening_over_last_2s_before_current_turn_event"]

        if i == 6:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_darkening_over_last_1s_before_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_brightening_over_last_1s_before_current_turn_event"]

        if i == 7:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_medium_darkening_since_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_strong_darkening_since_previous_turn_event"]

        if i == 8:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_medium_brightening_since_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_strong_brightening_since_previous_turn_event"]


        for j in range(len(angle_change0)):

            x1 = np.random.random() * 0.2 - 0.1 + [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2]
            x2 = np.random.random() * 0.2 - 0.1 + [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2 + 1]
            y1 = angle_change0[j]
            y2 = angle_change1[j]

            myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=0.25, zorder=1, alpha=0.5)
            myfig.Scatter(p0, x=[x1], y=[y1], lc=color, pt='o', lw=0.25, ps=2, pc='white', zorder=2, alpha=0.5)
            myfig.Scatter(p0, x=[x2], y=[y2], lc=color, pt='o', lw=0.25, ps=2, pc='white', zorder=2, alpha=0.5)

        x1 = [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2]
        x2 = [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2 + 1]
        y1 = np.mean(angle_change0)
        y2 = np.mean(angle_change1)

        myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=1, zorder=3, alpha=0.9)
        myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=1, zorder=3, alpha=0.9)
        myfig.Scatter(p0, x=[x1, x2], y=[y1, y2], lc=color, pt='o', lw=0.25, ps=2, pc='white', zorder=4, alpha=0.9)

        p = ttest_1samp(angle_change0 - angle_change1, 0, nan_policy='omit')[1]
        print("Angle change statistical comparison", i, "Experiment", experiment_name, ": p = ", p, np.mean(angle_change0 - angle_change1))

        myfig.Line(p0, x=[x1 + 0.1, x2 - 0.1], y=[55, 55], lc='black', lw=0.75)
        if p < 0.001:
            myfig.Text(p0, x1 + 0.5, 60, "***")
        elif p < 0.01:
            myfig.Text(p0, x1 + 0.5, 60, "**")
        elif p < 0.05:
            myfig.Text(p0, x1 + 0.5, 60, "*")
        else:
            myfig.Text(p0, x1 + 0.5, 60, "ns")

##########
for experiment_name, x_pos, y_pos, color in [["virtual_valley_stimulus_drosolarva", 2, 11, 'C0'],
                                             ["virtual_valley_stimulus_control_drosolarva", 2, 9, "gray"]]:

    p0 = myfig.Plot(fig, num='b', xpos=x_pos, ypos=y_pos, plot_height=1.25, plot_width=0.375*24,
                    lw=1, pc='white', errorbar_area=False,
                    xl="", xmin=-0.5, xmax=23.5, xticks=[0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23],
                    xticklabels=["Dark at current turn event",
                                 "Bright at current turn event",
                                 "Dark at previous turn event",
                                 "Bright at previous turn event",

                                 "Darkening since previous turn event",
                                 "Brightening since previous turn event",
                                 "Darkening over last 10 s before current turn event",
                                 "Brightening over last 10 s before current turn event",
                                 "Darkening over last 5 s before current turn event",
                                 "Brightening over last 5 s before current turn event",
                                 "Darkening over last 2 s before current turn event",
                                 "Brightening over last 2 s before current turn event",
                                 "Darkening over last 1 s before current turn event",
                                 "Brightening over last 1 s before current turn event",

                                 "Medium darkening since last turn event",
                                 "Strong darkening since last turn event",
                                 "Medium brightening since last turn event",
                                 "Strong brightening since last turn event"] if experiment_name == "virtual_valley_stimulus_control_drosolarva" else [""]*18,
                    xticklabels_rotation=45,
                    yl="Time since previous turn event (s)", ymin=-1, ymax=51, yticks=[0, 25, 50])

    for i in range(9):
        if i == 0:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_dark_at_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_bright_at_current_turn_event"]

        if i == 1:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_dark_at_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_bright_at_previous_turn_event"]

        if i == 2:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_darkening_since_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_brightening_since_previous_turn_event"]

        if i == 3:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_10s_before_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_10s_before_current_turn_event"]

        if i == 4:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_5s_before_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_5s_before_current_turn_event"]

        if i == 5:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_2s_before_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_2s_before_current_turn_event"]

        if i == 6:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_darkening_over_last_1s_before_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_brightening_over_last_1s_before_current_turn_event"]

        if i == 7:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_medium_darkening_since_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_strong_darkening_since_previous_turn_event"]

        if i == 8:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_medium_brightening_since_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_strong_brightening_since_previous_turn_event"]


        for j in range(len(angle_change0)):

            x1 = np.random.random() * 0.2 - 0.1 + [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2]
            x2 = np.random.random() * 0.2 - 0.1 + [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2 + 1]
            y1 = angle_change0[j]
            y2 = angle_change1[j]

            myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=0.25, zorder=1, alpha=0.5)
            myfig.Scatter(p0, x=[x1], y=[y1], lc=color, pt='o', lw=0.25, ps=1, pc='white', zorder=2, alpha=0.5)
            myfig.Scatter(p0, x=[x2], y=[y2], lc=color, pt='o', lw=0.25, ps=1, pc='white', zorder=2, alpha=0.5)

        x1 = [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2]
        x2 = [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2 + 1]
        y1 = np.mean(angle_change0)
        y2 = np.mean(angle_change1)

        myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=1, zorder=3, alpha=0.9)
        myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=1, zorder=3, alpha=0.9)
        myfig.Scatter(p0, x=[x1, x2], y=[y1, y2], lc=color, pt='o', lw=0.25, ps=2, pc='white', zorder=4, alpha=0.9)

        p = ttest_1samp(angle_change0 - angle_change1, 0, nan_policy='omit')[1]
        print("Run length statistical comparison", i, "Experiment", experiment_name, ": p = ", p, np.mean(angle_change0 - angle_change1))

        myfig.Line(p0, x=[x1 + 0.1, x2 - 0.1], y=[50, 50], lc='black', lw=0.75)
        if p < 0.001:
            myfig.Text(p0, x1 + 0.5, 55, "***")
        elif p < 0.01:
            myfig.Text(p0, x1 + 0.5, 55, "**")
        elif p < 0.05:
            myfig.Text(p0, x1 + 0.5, 55, "*")
        else:
            myfig.Text(p0, x1 + 0.5, 55, "ns")


fig.savepdf(root_path / f"figure2", open_pdf=True)