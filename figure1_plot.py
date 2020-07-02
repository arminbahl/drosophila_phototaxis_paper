import pandas as pd
from pathlib import Path
import pylab as pl
import my_figure as myfig
from scipy.stats import ttest_ind, ttest_1samp
import numpy as np

root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")
#df1 = pd.read_hdf(root_path / "all_data_deepposekit.h5", key="results_figure1")
#df1 = pd.read_hdf(root_path / "all_data_model_profile1.h5", key="results_figure1")
df1 = pd.read_hdf(root_path / "all_data_model_profile2.h5", key="results_figure1")

#df1.to_excel(root_path / "all_data_figure1.xlsx", sheet_name="results_figure1")
df1.groupby(level=[0, 2, 3]).mean().to_excel(root_path / "all_data_figure1_experiment_mean.xlsx", sheet_name="results_figure1")

df2 = pd.read_hdf(root_path / "all_data_spatial_phototaxis.h5", key="results_figure1")
df2.to_excel(root_path / "all_data__spatial_phototaxis_figure1.xlsx", sheet_name="results_figure1")
df2.groupby(level=[0, 2, 3]).mean().to_excel(root_path / "all_data_spatial_phototaxis_figure1_experiment_mean.xlsx", sheet_name="results_figure1")

# We define the phototaxis index as the change in time spent in the dark ring relative to control conditions

fraction_of_time_in_dark_ring_experiment = df1.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and region_bin == 'r2_to_r4' and time_bin == 't15_to_t60'")[f"fraction_of_time_spent"]
#fraction_of_time_in_dark_ring_experiment = df1.query("experiment_name == 'temporal_phototaxis_drosolarva' and region_bin == 'r2_to_r4' and time_bin == 't15_to_t60'")[f"fraction_of_time_spent"]

fraction_of_time_in_dark_ring_control = df1.query("experiment_name == 'virtual_valley_stimulus_control_drosolarva' and region_bin == 'r2_to_r4' and time_bin == 't15_to_t60'")[f"fraction_of_time_spent"]
#
# ds = []
# for i in range(1000):
#     a = np.random.choice(fraction_of_time_in_dark_ring_experiment, 100).mean()
#     b = np.random.choice(fraction_of_time_in_dark_ring_control, 100).mean()
#     ds.append( (a-b))
#
# phototaxis_index_mean = np.mean(ds)
# print(phototaxis_index_mean)

for experiment_name in ["virtual_valley_stimulus_drosolarva", "temporal_phototaxis_drosolarva"]:

    if experiment_name == "virtual_valley_stimulus_drosolarva":
        color = 'C0'
        df = df1

    elif experiment_name == "temporal_phototaxis_drosolarva":
        color = "C1"
        df = df1
    else:
        color = "C2"
        df = df2

    fig = myfig.Figure(title=f"Figure 1 - {experiment_name}")

    for y_pos, time_bin in enumerate(["t15_to_t60", "t15_to_t25", "t25_to_t35", "t35_to_t45", "t45_to_t55"]):
        p0 = myfig.Plot(fig, num='a' if y_pos == 0 else '', xpos=1.5, ypos=22 - y_pos*3, plot_height=1.25, plot_width=2.0,
                            lw=1, pc='white', errorbar_area=False, title=f"Time bin: {time_bin}",
                            xl="", xmin=-0.5, xmax=2.5, xticks=[0, 1, 2], xticklabels=["Bright center", "Dark ring", "Bright ring"], xticklabels_rotation=45,
                            yl="Fraction of\ntime spent (%)", ymin=-5, ymax=105, yticks=[0, 50, 100])

        p1 = myfig.Plot(fig, num='b' if y_pos == 0 else '', xpos=5.5, ypos=22 - y_pos*3, plot_height=1.25, plot_width=2.0,
                            lw=1, pc='white', errorbar_area=False, title=f"Time bin: {time_bin}",
                            xl="", xmin=-0.5, xmax=2.5, xticks=[0, 1, 2], xticklabels=["Bright center", "Dark ring", "Bright ring"], xticklabels_rotation=45,
                            yl="Speed (cm/s)", ymin=-0.01, ymax=0.1, yticks=[0.0, 0.05, 0.1])

        for x_pos, region_bin in enumerate(["r0_to_r2", "r2_to_r4", "r4_to_r6"]):

            fraction_of_time_spent_experiment = df.query("experiment_name == @experiment_name and region_bin == @region_bin and time_bin == @time_bin")[f"fraction_of_time_spent"]
            fraction_of_time_spent_control = df1.query("experiment_name == 'virtual_valley_stimulus_control_drosolarva' and region_bin == @region_bin and time_bin == @time_bin")[f"fraction_of_time_spent"]

            p_time_spent = ttest_ind(fraction_of_time_spent_experiment, fraction_of_time_spent_control, nan_policy='omit', equal_var=False)[1]

            myfig.Scatter(p0, x=np.random.random(len(fraction_of_time_spent_experiment))*0.1 - 0.05  + x_pos - 0.2,
                          y=fraction_of_time_spent_experiment,
                          lc=color, pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)
            myfig.Scatter(p0, x=np.random.random(len(fraction_of_time_spent_control))*0.1 - 0.05 + x_pos + 0.2,
                          y=fraction_of_time_spent_control,
                          lc='gray', pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)

            myfig.Scatter(p0, x=[x_pos - 0.2],
                          y=[fraction_of_time_spent_experiment.mean()],
                          yerr=[fraction_of_time_spent_experiment.sem()],
                          lc=[color], pt='o', lw=0.5, ps=10, pc='white', zorder=3)
            myfig.Scatter(p0, x=[x_pos + 0.2],
                          y=[fraction_of_time_spent_control.mean()],
                          yerr=[fraction_of_time_spent_control.sem()],
                          lc=['gray'], pt='o', lw=0.5, ps=10, pc='white', zorder=3)

            y = [50, 70, 95][x_pos]
            myfig.Line(p0, x=[x_pos - 0.2, x_pos + 0.2], y=[y, y], lc='black', lw=0.75)
            if p_time_spent < 0.001:
                myfig.Text(p0, x_pos, y + 5, "***")
            elif p_time_spent < 0.01:
                myfig.Text(p0, x_pos, y + 5, "**")
            elif p_time_spent < 0.05:
                myfig.Text(p0, x_pos, y + 5, "*")
            else:
                myfig.Text(p0, x_pos, y + 10, "ns")

            ######## Speed
            print(region_bin, time_bin)
            speed_experiment = df.query("experiment_name == @experiment_name and region_bin == @region_bin and time_bin == @time_bin")[f"speed"]
            speed_control = df1.query("experiment_name == 'virtual_valley_stimulus_control_drosolarva' and region_bin == @region_bin and time_bin == @time_bin")[f"speed"]

            myfig.Scatter(p1, x=np.random.random(len(speed_experiment))*0.1 - 0.05 + x_pos - 0.2,
                          y=speed_experiment,
                          lc=color, pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)
            myfig.Scatter(p1, x=np.random.random(len(speed_control))*0.1 - 0.05 + x_pos + 0.2,
                          y=speed_control,
                          lc='gray', pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)

            myfig.Scatter(p1, x=[x_pos - 0.2],
                          y=[speed_experiment.mean()],
                          yerr=[speed_experiment.sem()],
                          lc=[color], pt='o', lw=0.5, ps=10, pc='white', zorder=3, label="Valley stimulus" if x_pos == 0 else None)
            myfig.Scatter(p1, x=[x_pos + 0.2],
                          y=[speed_control.mean()],
                          yerr=[speed_control.sem()],
                          lc=['gray'], pt='o', lw=0.5, ps=10, pc='white', zorder=3, label="Control (always gray)" if x_pos == 0 else None)

            p_speed = ttest_ind(speed_experiment, speed_control, nan_policy='omit', equal_var=False)[1]

            y = [0.07, 0.07, 0.07][x_pos]
            myfig.Line(p1, x=[x_pos - 0.2, x_pos + 0.2], y=[y, y], lc='black', lw=0.75)
            if p_speed < 0.001:
                myfig.Text(p1, x_pos, y + 0.005, "***")
            elif p_speed < 0.01:
                myfig.Text(p1, x_pos, y + 0.005, "**")
            elif p_speed < 0.05:
                myfig.Text(p1, x_pos, y + 0.005, "*")
            else:
                myfig.Text(p1, x_pos, y + 0.01, "ns")


            print(experiment_name, x_pos, p_time_spent,p_speed, len(speed_experiment), len(speed_control) )


    p0 = myfig.Plot(fig, num='c', xpos=1.5, ypos=7, plot_height=1.25, plot_width=4,
                            lw=1, pc='white', errorbar_area=False, title=f"Time bin: t15_to_t60",
                            xl="Distance to center (cm)", xmin=0, xmax=6, xticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                            yl="Fraction of\ntime spent (%)", ymin=-5, ymax=105, yticks=[0, 50, 100])


    # Zoom in with higher bin resolution

    for x_pos, region_bin in enumerate(["r0_to_r1", "r1_to_r2", "r2_to_r3", "r3_to_r4", "r4_to_r5", "r5_to_r6"]):
        fraction_of_time_spent_experiment = df.query("experiment_name == @experiment_name and region_bin == @region_bin and time_bin == 't15_to_t60'")[f"fraction_of_time_spent"]
        fraction_of_time_spent_control = df1.query("experiment_name == 'virtual_valley_stimulus_control_drosolarva' and region_bin == @region_bin and time_bin == 't15_to_t60'")[f"fraction_of_time_spent"]

        p_time_spent = ttest_ind(fraction_of_time_spent_experiment, fraction_of_time_spent_control, nan_policy='omit', equal_var=False)[1]
        print(experiment_name, region_bin, p_time_spent)

        myfig.Scatter(p0, x=np.random.random(len(fraction_of_time_spent_experiment))*0.1 - 0.05 + x_pos - 0.2 + 0.5,
                      y=fraction_of_time_spent_experiment, label="Valley stimulus" if x_pos == 0 else None,
                      lc=color, pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)
        myfig.Scatter(p0, x=np.random.random(len(fraction_of_time_spent_control))*0.1 - 0.05 + x_pos + 0.2 + 0.5,
                      y=fraction_of_time_spent_control, label="Control (always gray)" if x_pos == 0 else None,
                      lc='gray', pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)

        myfig.Scatter(p0, x=[x_pos - 0.2 + 0.5],
                      y=[fraction_of_time_spent_experiment.mean()],
                      yerr=[fraction_of_time_spent_experiment.sem()],
                      lc=[color], pt='o', lw=0.5, ps=10, pc='white', zorder=3)
        myfig.Scatter(p0, x=[x_pos + 0.2 + 0.5],
                      y=[fraction_of_time_spent_control.mean()],
                      yerr=[fraction_of_time_spent_control.sem()],
                      lc=['gray'], pt='o', lw=0.5, ps=10, pc='white', zorder=3)


        y = [50, 70, 70, 70, 70, 85][x_pos]
        myfig.Line(p0, x=[x_pos + 0.5 - 0.2, x_pos + 0.5 + 0.2], y=[y, y], lc='black', lw=0.75)
        if p_time_spent < 0.001:
            myfig.Text(p0, x_pos + 0.5, y + 5, "***")
        elif p_time_spent < 0.01:
            myfig.Text(p0, x_pos + 0.5, y + 5, "**")
        elif p_time_spent < 0.05:
            myfig.Text(p0, x_pos + 0.5, y + 5, "*")
        else:
            myfig.Text(p0, x_pos + 0.5, y + 10, "ns")



    fig.savepdf(root_path / f"figure1_{experiment_name}", open_pdf=True)