import pandas as pd
from pathlib import Path
import pylab as pl
import my_figure as myfig
from scipy.stats import ttest_ind, ttest_1samp
import numpy as np
from sklearn.linear_model import LinearRegression

root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")

# df = pd.read_hdf(root_path / "all_events.h5", key="results_figure2")
# df_histogram_results = pd.read_hdf(root_path / "all_events.h5", key="results_figure2_histograms")
# df_event_triggered_luminance = pd.read_hdf(root_path / "all_events.h5", key="event_triggered_luminance")

df = pd.read_hdf(root_path / "all_events_model_profile1.h5", key="results_figure2")
df_histogram_results = pd.read_hdf(root_path / "all_events_model_profile1.h5", key="results_figure2_histograms")
df_event_triggered_luminance = pd.read_hdf(root_path / "all_events_model_profile1.h5", key="event_triggered_luminance")

#df.to_excel(root_path / "all_events_figure2.xlsx", sheet_name="all_events_model.h5")
#df.groupby("experiment_name").mean().to_excel(root_path / "all_events_model_figure2_experiment_mean.xlsx", sheet_name="all_data")


fig = myfig.Figure(title="Figure 2")

##########
p0 = myfig.Plot(fig, num='a', xpos=1.5, ypos=22, plot_height=0.75, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Turn angle (deg)", xmin=-181, xmax=181, xticks=[-180, -90, 0, 90, 180],
                    yl="Probability", ymin=-0.001, ymax=0.012)

df_selected = df_histogram_results.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and histogram_type == 'angle_change'").reset_index(level=['experiment_name', 'histogram_type'], drop=True)
myfig.Bar(p0, x=df_selected.index, y=df_selected["density"].values, lc='C0', lw=0, width=0.95*362/60)

##########
p0 = myfig.Plot(fig, num='b', xpos=4.5, ypos=22, plot_height=0.75, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Run length (s)", xmin=-0.5, xmax=60.5, xticks=[0, 30, 60],
                    yl="Probability", ymin=-0.001, ymax=0.06)

df_selected = df_histogram_results.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and histogram_type == 'run_length'").reset_index(level=['experiment_name', 'histogram_type'], drop=True)
myfig.Bar(p0, x=df_selected.index, y=df_selected["density"].values, lc='C0', lw=0, width=0.95*61/60)

##########
df_selected = df_histogram_results.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and histogram_type == 'luminance_change_since_previous_turn_event'").reset_index(level=['experiment_name', 'histogram_type'], drop=True)

p0 = myfig.Plot(fig, num='b', xpos=7.5, ypos=22, plot_height=0.75, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Brightness change during runs", xmin=-81, xmax=81, xticks=[-80, -40, 0, 40, 80],
                    yl="Probability", ymin=-0.001, ymax=df_selected["density"].values.max())

myfig.Bar(p0, x=df_selected.index, y=df_selected["density"].values, lc='C0', lw=0, width=0.95*162/60)

##########
df_selected = df_histogram_results.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and histogram_type == 'luminance_change_during_current_turn_event'").reset_index(level=['experiment_name', 'histogram_type'], drop=True)

p0 = myfig.Plot(fig, num='b', xpos=10.5, ypos=22, plot_height=0.75, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Brightness change during turns", xmin=-81, xmax=81, xticks=[-80, -40, 0, 40, 80],
                    yl="Probability", ymin=-0.001, ymax=df_selected["density"].values.max())

myfig.Bar(p0, x=df_selected.index, y=df_selected["density"].values, lc='C0', lw=0, width=0.95*162/60)


##########
p0 = myfig.Plot(fig, num='e', xpos=14, ypos=22, plot_height=2, plot_width=1.5,
                        lw=1, pc='white', errorbar_area=True, hlines=[0], vlines=[0],
                        xl="Time relative to turn event (s)", xmin=-21.5, xmax=21.5, xticks=[-20, -10, 0, 10, 20],
                        yl="Brightness relative to turn event", ymin=-21, ymax=6, yticks=[-20, -10, 0])

myfig.Line(p0, x=df_event_triggered_luminance.index,
           y=df_event_triggered_luminance.means_experiment,
           yerr=df_event_triggered_luminance.sems_experiment,
           lc='C0', lw=0.5, zorder=1)

myfig.Line(p0, x=df_event_triggered_luminance.index,
           y=df_event_triggered_luminance.means_control,
           yerr=df_event_triggered_luminance.sems_control,
           lc='gray', lw=0.5, zorder=1)



##########
for experiment_name, x_pos, y_pos, color in [["virtual_valley_stimulus_drosolarva", 2, 19, "C0"],
                                             ["virtual_valley_stimulus_control_drosolarva", 2, 17, "gray"]]:

    p0 = myfig.Plot(fig, num='b', xpos=x_pos, ypos=y_pos, plot_height=1.25, plot_width=0.375*24,
                    lw=1, pc='white', errorbar_area=False,
                    xl="", xmin=-0.5, xmax=23.5, xticks=[0, 1, 2.5, 3.5],
                    xticklabels=["Dark at current turn event",
                                 "Bright at current turn event",

                                 "Darkening since previous turn event",
                                 "Brightening since previous turn event"] if experiment_name == "virtual_valley_stimulus_control_drosolarva" else [""]*4,
                    xticklabels_rotation=45,
                    yl="Absolute angle change (°)", ymin=-5, ymax=65, yticks=[0, 30, 60])

    for i in range(2):
        if i == 0:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_dark_at_current_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_bright_at_current_turn_event"]

        if i == 1:
            angle_change0 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_darkening_since_previous_turn_event"]
            angle_change1 = df.query("experiment_name == @experiment_name")[f"angle_change_at_current_turn_event_if_brightening_since_previous_turn_event"]

        for j in range(len(angle_change0)):

            x1 = np.random.random() * 0.2 - 0.1 + [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2]
            x2 = np.random.random() * 0.2 - 0.1 + [0, 1, 2.5, 3.5, 6, 7, 8.5, 9.5, 11, 12, 13.5, 14.5, 16, 17, 19.5, 20.5, 22, 23][i * 2 + 1]
            y1 = angle_change0[j]
            y2 = angle_change1[j]

            myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=0.25, zorder=1, alpha=0.5)
            myfig.Scatter(p0, x=[x1], y=[y1], lc=color, pt='o', lw=0.25, ps=2, pc='white', zorder=2, alpha=0.5)
            myfig.Scatter(p0, x=[x2], y=[y2], lc=color, pt='o', lw=0.25, ps=2, pc='white', zorder=2, alpha=0.5)

        x1 = [0, 1, 2.5, 3.5][i * 2]
        x2 = [0, 1, 2.5, 3.5][i * 2 + 1]
        y1 = np.mean(angle_change0)
        y2 = np.mean(angle_change1)

        myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=1, zorder=3, alpha=0.9)
        myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=1, zorder=3, alpha=0.9)
        myfig.Scatter(p0, x=[x1, x2], y=[y1, y2], lc=color, pt='o', lw=0.25, ps=2, pc='white', zorder=4, alpha=0.9)

        p = ttest_1samp(angle_change0 - angle_change1, 0, nan_policy='omit')[1]
        print("Angle change statistical comparison", i, "Experiment", experiment_name, ": p = ", p, np.mean(angle_change0 - angle_change1), "n = ", len(angle_change0 - angle_change1))

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
                    xl="", xmin=-0.5, xmax=23.5, xticks=[0, 1, 2.5, 3.5],
                    xticklabels=["Dark at current turn event",
                                 "Bright at current turn event",

                                 "Darkening since previous turn event",
                                 "Brightening since previous turn event",
                                 ] if experiment_name == "virtual_valley_stimulus_control_drosolarva" else [""]*4,
                    xticklabels_rotation=45,
                    yl="Time since previous turn event (s)", ymin=-1, ymax=51, yticks=[0, 25, 50])

    for i in range(2):
        if i == 0:
            run_length0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_dark_at_current_turn_event"]
            run_length1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_bright_at_current_turn_event"]

        if i == 1:
            run_length0 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_darkening_since_previous_turn_event"]
            run_length1 = df.query("experiment_name == @experiment_name")[f"time_since_previous_turn_event_at_current_turn_event_if_brightening_since_previous_turn_event"]

        for j in range(len(angle_change0)):

            x1 = np.random.random() * 0.2 - 0.1 + [0, 1, 2.5, 3.5][i * 2]
            x2 = np.random.random() * 0.2 - 0.1 + [0, 1, 2.5, 3.5][i * 2 + 1]
            y1 = run_length0[j]
            y2 = run_length1[j]

            myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=0.25, zorder=1, alpha=0.5)
            myfig.Scatter(p0, x=[x1], y=[y1], lc=color, pt='o', lw=0.25, ps=1, pc='white', zorder=2, alpha=0.5)
            myfig.Scatter(p0, x=[x2], y=[y2], lc=color, pt='o', lw=0.25, ps=1, pc='white', zorder=2, alpha=0.5)

        x1 = [0, 1, 2.5, 3.5][i * 2]
        x2 = [0, 1, 2.5, 3.5][i * 2 + 1]
        y1 = np.mean(run_length0)
        y2 = np.mean(run_length1)

        myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=1, zorder=3, alpha=0.9)
        myfig.Line(p0, x=[x1, x2], y=[y1, y2], lc=color, lw=1, zorder=3, alpha=0.9)
        myfig.Scatter(p0, x=[x1, x2], y=[y1, y2], lc=color, pt='o', lw=0.25, ps=2, pc='white', zorder=4, alpha=0.9)

        p = ttest_1samp(run_length0 - run_length1, 0, nan_policy='omit')[1]
        print("Run length statistical comparison", i, "Experiment", experiment_name, ": p = ", p, np.mean(run_length0 - run_length1), "n = ", len(run_length0 - run_length1))

        myfig.Line(p0, x=[x1 + 0.1, x2 - 0.1], y=[50, 50], lc='black', lw=0.75)
        if p < 0.001:
            myfig.Text(p0, x1 + 0.5, 55, "***")
        elif p < 0.01:
            myfig.Text(p0, x1 + 0.5, 55, "**")
        elif p < 0.05:
            myfig.Text(p0, x1 + 0.5, 55, "*")
        else:
            myfig.Text(p0, x1 + 0.5, 55, "ns")



# Luminance change
##########
for experiment_name, x_pos, y_pos, color in [["virtual_valley_stimulus_drosolarva", 19, 19, "C0"],
                                             ["virtual_valley_stimulus_control_drosolarva", 19, 15, "gray"]]:
    p0 = myfig.Plot(fig, num='e', xpos=x_pos, ypos=y_pos, plot_height=1.25, plot_width=0.375*3,
                        lw=1, pc='white', errorbar_area=False,
                        xl="", xmin=-0.5, xmax=2.5, xticks=[0, 1, 2], xticklabels=["Since previous turn event",
                                                                                   "During turn event", "kk"], xticklabels_rotation=45,
                        yl="Absolute brightness change", ymin=-5, ymax=65, yticks=[0, 30, 60])


    for j in range(len(df.query("experiment_name == @experiment_name"))):

        for i in range(2):
            if i == 0:
                luminance_change = df.query("experiment_name == @experiment_name")[f"luminance_change_since_previous_turn_event"]
            if i == 1:
                luminance_change = df.query("experiment_name == @experiment_name")[f"luminance_change_during_current_turn_event"]

            x = np.random.random() * 0.2 - 0.1 + i
            y = luminance_change[j]

            myfig.Scatter(p0, x=[x], y=[y], lc=color, pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)

            if i > 0:
                myfig.Line(p0, x=[x, previous_x], y=[y, previous_y], lc=color, lw=0.5, zorder=1, alpha=0.5)

            previous_x = x
            previous_y = y

    for i in range(1):
        if i == 0:
            p = ttest_1samp(df.query("experiment_name == @experiment_name")[f"luminance_change_since_previous_turn_event"] -
                            df.query("experiment_name == @experiment_name")[f"luminance_change_during_current_turn_event"], 0, nan_policy='omit')[1]

        myfig.Line(p0, x=[0.1 + i, 0.9 + i], y=[62, 62], lc='black', lw=0.75)
        if p < 0.001:
            myfig.Text(p0, 0.5 + i, 65, "***")
        elif p < 0.01:
            myfig.Text(p0, 0.5 + i, 65, "**")
        elif p < 0.05:
            myfig.Text(p0, 0.5 + i, 65, "*")
        else:
            myfig.Text(p0, 0.5 + i, 65, "ns")


# The individual event analsis
df = pd.read_hdf(root_path / "all_events_model_profile1.h5", key="all_events")
#df = pd.read_hdf(root_path / "all_events.h5", key="all_events")

for experiment_name in ["virtual_valley_stimulus_drosolarva", "virtual_valley_stimulus_control_drosolarva"]:

    if experiment_name == "virtual_valley_stimulus_drosolarva":
        ypos = 8
        color = 'C0'
    else:
        ypos = 3
        color = 'gray'

    df_selected = df.query("experiment_name == @experiment_name and time_at_current_turn_event > 15*60 and time_at_current_turn_event <= 60*60")
    df_selected.loc[:, "r_at_previous_turn_event"] = df_selected["r_at_current_turn_event"].shift(1).copy()
    df_selected.loc[:, "r_at_next_turn_event"] = df_selected["r_at_current_turn_event"].shift(-1).copy()
    df_selected = df_selected.query("r_at_current_turn_event < 5.9 and r_at_previous_turn_event < 5.9 and r_at_next_turn_event < 5.9")

    p0 = myfig.Plot(fig, num='b', xpos=10, ypos=ypos, plot_height=1.25, plot_width=2,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Brightness", xmin=-5, xmax=181, xticks=[0, 90, 180],
                    yl="Absolute turn angle", ymin=-1, ymax=41, yticks=[0, 20, 40])

    df_selected1 = df_selected.query("angle_change_at_current_turn_event < 41 and "
                                     "angle_change_at_current_turn_event > -41 and "
                                     "luminance_at_current_turn_event < 181")[["luminance_at_current_turn_event",
                                                                               "angle_change_at_current_turn_event"]]



    myfig.Scatter(p0, x=df_selected1["luminance_at_current_turn_event"],
                  y=df_selected1["angle_change_at_current_turn_event"].abs(),
                  lc=None, lw=0, pt='.', ps=1, pc=color, zorder=4, alpha=0.3)

    bins = np.arange(0, 161, 40)
    vals = []
    for bin in bins:
        df_ = df_selected1.query("luminance_at_current_turn_event < (@bin + 20) and luminance_at_current_turn_event > (@bin - 20)")
        vals.append(df_["angle_change_at_current_turn_event"].abs().median())
    myfig.Scatter(p0, x=bins, y=vals, lw=0, pt='.', ps=12, pc=color, zorder=5)

    X_median = np.array(bins).reshape(-1, 1)
    Y_median = np.array(vals).reshape(-1, 1)

    X_raw = np.array(df_selected1["luminance_at_current_turn_event"]).reshape(-1, 1)
    Y_raw = np.array(df_selected1["angle_change_at_current_turn_event"].abs()).reshape(-1, 1)

    linear_regressor_median = LinearRegression()  # create object for the class
    reg_median = linear_regressor_median.fit(X_median, Y_median)  # perform linear regression
    Y_pred_median = linear_regressor_median.predict(X_median)  # make predictions
    real_R_median = reg_median.score(X_median, Y_median)

    linear_regressor_raw = LinearRegression()  # create object for the class
    reg_raw = linear_regressor_raw.fit(X_raw, Y_raw)  # perform linear regression
    Y_pred_raw = linear_regressor_raw.predict(X_raw)  # make predictions
    real_R_raw = reg_raw.score(X_raw, Y_raw)


    # Repeat this but with shuffled data
    Rs_shuffled_raw = []
    Rs_shuffled_median = []

    for shuffle_i in range(1000):

        df_selected_shuffled = df_selected1.apply(lambda df_selected1: df_selected1.sample(frac=1).values)

        bins = np.arange(0, 161, 40)
        vals = []
        for bin in bins:
            df_ = df_selected_shuffled.query(
                "luminance_at_current_turn_event < (@bin + 20) and luminance_at_current_turn_event > (@bin - 20)")
            vals.append(df_["angle_change_at_current_turn_event"].abs().median())

        X_shuffled_median = np.array(bins).reshape(-1, 1)
        Y_shuffled_median = np.array(vals).reshape(-1, 1)

        X_shuffled_raw = np.array(df_selected_shuffled["luminance_at_current_turn_event"]).reshape(-1, 1)
        Y_shuffled_raw = np.array(df_selected_shuffled["angle_change_at_current_turn_event"].abs()).reshape(-1, 1)

        linear_regressor_shuffled_median = LinearRegression()  # create object for the class
        reg_shuffled_median = linear_regressor_shuffled_median.fit(X_shuffled_median, Y_shuffled_median)  # perform linear regression
        Y_pred_shuffled_median = linear_regressor_shuffled_median.predict(X_shuffled_median)  # make predictions
        real_R_shuffled_median = reg_shuffled_median.score(X_shuffled_median, Y_shuffled_median)

        linear_regressor_shuffled_raw = LinearRegression()  # create object for the class
        reg_shuffled_raw = linear_regressor_shuffled_raw.fit(X_shuffled_raw, Y_shuffled_raw)  # perform linear regression
        Y_pred_shuffled_raw = linear_regressor_shuffled_raw.predict(X_shuffled_raw)  # make predictions
        real_R_shuffled_raw = reg_raw.score(X_shuffled_raw, Y_shuffled_raw)

        Rs_shuffled_median.append(real_R_shuffled_median)
        Rs_shuffled_raw.append(real_R_shuffled_raw)

    Rs_shuffled_median = np.array(Rs_shuffled_median)
    Rs_shuffled_raw = np.array(Rs_shuffled_raw)

    p_median = np.sum(Rs_shuffled_median > real_R_median)/len(Rs_shuffled_median)
    p_raw = np.sum(Rs_shuffled_raw > real_R_raw) / len(Rs_shuffled_raw)

    myfig.Line(p0, x=np.array([0, 180]),
               y=reg_median.coef_[0][0]*np.array([0, 180]) + reg_median.intercept_[0],
               lc=color, lw=0.5, zorder=6, label=f'R2 = {real_R_median:.3f}, p = {p_median:.3f}\ny = {reg_median.coef_[0][0]:.3f}*x + {reg_median.intercept_[0]:.2f}')
    myfig.Line(p0, x=np.array([0, 180]),
               y=reg_raw.coef_[0][0]*np.array([0, 180]) + reg_raw.intercept_[0],
               lc=color, dashes=(2,2), lw=0.5, zorder=6, label=f'R2 = {real_R_raw:.3f}, p = {p_raw:.3f}\ny = {reg_raw.coef_[0][0]:.3f}*x + {reg_raw.intercept_[0]:.2f}')



    #######
    # The luminance change
    p0 = myfig.Plot(fig, num='b', xpos=16, ypos=ypos, plot_height=1.25, plot_width=2,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Brightness change\nsince previous turn", xmin=-65, xmax=65, xticks=[-60, -30, 0, 30, 60],
                    yl="Absolute turn angle", ymin=-1, ymax=41, yticks=[0, 20, 40])

    df_selected1 = df_selected.query("luminance_change_since_previous_turn_event > -61 and "
                                     "luminance_change_since_previous_turn_event < 61 and "
                                     "angle_change_at_current_turn_event < 41 and "
                                     "angle_change_at_current_turn_event > -41")[["luminance_change_since_previous_turn_event",
                                                                               "angle_change_at_current_turn_event"]]

    linear_regressor = LinearRegression()  # create object for the class

    myfig.Scatter(p0, x=df_selected1["luminance_change_since_previous_turn_event"],
                  y=df_selected1["angle_change_at_current_turn_event"].abs(),
                  lc=None, lw=0, pt='.', ps=1, pc=color, zorder=4, alpha=0.3)

    bins = np.arange(-60, 61, 30)
    vals = []
    for bin in bins:
        df_ = df_selected1.query("luminance_change_since_previous_turn_event < (@bin + 15) and luminance_change_since_previous_turn_event > (@bin - 15)")
        vals.append(df_["angle_change_at_current_turn_event"].abs().median())
    myfig.Scatter(p0, x=bins, y=vals, lw=0, pt='.', ps=12, pc=color, zorder=5)

    X_median = np.array(bins).reshape(-1, 1)
    Y_median = np.array(vals).reshape(-1, 1)

    X_raw = np.array(df_selected1["luminance_change_since_previous_turn_event"]).reshape(-1, 1)
    Y_raw = np.array(df_selected1["angle_change_at_current_turn_event"].abs()).reshape(-1, 1)

    linear_regressor_median = LinearRegression()  # create object for the class
    reg_median = linear_regressor_median.fit(X_median, Y_median)  # perform linear regression
    Y_pred_median = linear_regressor_median.predict(X_median)  # make predictions
    real_R_median = reg_median.score(X_median, Y_median)

    linear_regressor_raw = LinearRegression()  # create object for the class
    reg_raw = linear_regressor_raw.fit(X_raw, Y_raw)  # perform linear regression
    Y_pred_raw = linear_regressor_raw.predict(X_raw)  # make predictions
    real_R_raw = reg_raw.score(X_raw, Y_raw)

    # Repeat this but with shuffled data
    Rs_shuffled_raw = []
    Rs_shuffled_median = []

    for shuffle_i in range(1000):

        df_selected_shuffled = df_selected1.apply(lambda df_selected1: df_selected1.sample(frac=1).values)

        bins = np.arange(-60, 61, 30)
        vals = []
        for bin in bins:
            df_ = df_selected_shuffled.query("luminance_change_since_previous_turn_event < (@bin + 15) and luminance_change_since_previous_turn_event > (@bin - 15)")
            vals.append(df_["angle_change_at_current_turn_event"].abs().median())

        X_shuffled_median = np.array(bins).reshape(-1, 1)
        Y_shuffled_median = np.array(vals).reshape(-1, 1)

        X_shuffled_raw = np.array(df_selected_shuffled["luminance_change_since_previous_turn_event"]).reshape(-1, 1)
        Y_shuffled_raw = np.array(df_selected_shuffled["angle_change_at_current_turn_event"].abs()).reshape(-1, 1)

        linear_regressor_shuffled_median = LinearRegression()  # create object for the class
        reg_shuffled_median = linear_regressor_shuffled_median.fit(X_shuffled_median,
                                                                   Y_shuffled_median)  # perform linear regression
        Y_pred_shuffled_median = linear_regressor_shuffled_median.predict(X_shuffled_median)  # make predictions
        real_R_shuffled_median = reg_shuffled_median.score(X_shuffled_median, Y_shuffled_median)

        linear_regressor_shuffled_raw = LinearRegression()  # create object for the class
        reg_shuffled_raw = linear_regressor_shuffled_raw.fit(X_shuffled_raw,
                                                             Y_shuffled_raw)  # perform linear regression
        Y_pred_shuffled_raw = linear_regressor_shuffled_raw.predict(X_shuffled_raw)  # make predictions
        real_R_shuffled_raw = reg_raw.score(X_shuffled_raw, Y_shuffled_raw)

        Rs_shuffled_median.append(real_R_shuffled_median)
        Rs_shuffled_raw.append(real_R_shuffled_raw)

    Rs_shuffled_median = np.array(Rs_shuffled_median)
    Rs_shuffled_raw = np.array(Rs_shuffled_raw)

    p_median = np.sum(Rs_shuffled_median > real_R_median) / len(Rs_shuffled_median)
    p_raw = np.sum(Rs_shuffled_raw > real_R_raw) / len(Rs_shuffled_raw)

    myfig.Line(p0, x=np.array([-61, 61]),
               y=reg_median.coef_[0][0] * np.array([-61, 61]) + reg_median.intercept_[0],
               lc='black', lw=0.5, zorder=6,
               label=f'R2 = {real_R_median:.3f}, p = {p_median:.3f}\ny = {reg_median.coef_[0][0]:.3f}*x + {reg_median.intercept_[0]:.2f}')
    myfig.Line(p0, x=np.array([-61, 61]),
               y=reg_raw.coef_[0][0] * np.array([-61, 61]) + reg_raw.intercept_[0],
               lc='black', dashes=(2, 2), lw=0.5, zorder=6,
               label=f'R2 = {real_R_raw:.3f}, p = {p_raw:.3f}\ny = {reg_raw.coef_[0][0]:.3f}*x + {reg_raw.intercept_[0]:.2f}')

fig.savepdf(root_path / f"figure2_model_profile1", open_pdf=True)