import pandas as pd
from pathlib import Path
import numpy as np
import pylab as pl
import my_figure as myfig
import matplotlib

root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")
#df = pd.read_hdf(root_path / "all_data_deepposekit.h5", key="raw_data")
df = pd.read_hdf(root_path / "all_data_model_profile2.h5", key="raw_data")

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
              "larva_ID != '2018_11_20_fish035_setup0'")


all_results = dict({"experiment_name": [],
                    "larva_ID": [],
                    "region_bin": [],
                    "time_bin": [],
                    "fraction_of_time_spent": [],
                    "speed": []})

experiment_names = df.index.get_level_values('experiment_name').unique().values

fig = myfig.Figure(title="Figure 1 examples")

xs = np.arange(-6, 6, 0.025)
ys = np.arange(-6, 6, 0.025)

XX, YY = np.meshgrid(xs, ys)

r = np.sqrt(XX ** 2 + YY ** 2)
c1 = (1 - pow((1 - r/6), 0.5)) * 255
c2 = np.ones_like(r) * 200
c3 = 255*(r-3)**2/9

c1[np.where(r >= 5.6)] = np.nan
c2[np.where(r >= 5.6)] = np.nan
c3[np.where(r >= 5.6)] = np.nan

for i in range(3):

    experiment_name = ["temporal_phototaxis_drosolarva", "virtual_valley_stimulus_control_drosolarva", "virtual_valley_stimulus_drosolarva"][i]
    color = ['C1', "gray", "C0"][i]
    profile = [c1, c2, c3][i]

    df_selected = df.query("experiment_name == @experiment_name").reset_index(level=['experiment_name'], drop=True)
    larva_IDs = df_selected.index.get_level_values('larva_ID').unique().values

    p0 = myfig.Plot(fig, xpos=3 + 5*i, ypos=15, plot_height=3, plot_width=3, lw=1, title=experiment_name,
                    xl="X (cm)", xmin=-6.1, xmax=6.1, xticks=[-6, -3, 0, 3, 6],
                    yl="Y (cm)", ymin=-6.1, ymax=6.1, yticks=[-6, -3, 0, 3, 6])

    p0.ax.imshow(profile, extent=(-6, 6, -6, 6), interpolation='bilinear', aspect='auto', cmap='gray', vmin=0, vmax=255, alpha=0.65, zorder=1)

    for larva_ID in larva_IDs:
        print(larva_ID)
        df_selected_larva = df_selected.query("larva_ID == @larva_ID and time >= 15*60 and time < 60*60").reset_index(level=['larva_ID'], drop=True)[["r", "x", "y"]]
        df_selected_larva.loc[df_selected_larva.r > 5.5] = np.nan

        # Downsample the x,y positions to 1 s
        df_selected_larva.index = pd.to_datetime(df_selected_larva.index, unit='s')  # Convert seconds into datetime objects

        df_selected_larva = df_selected_larva.resample('1s').median()
        df_selected_larva.index = (df_selected_larva.index - pd.to_datetime(0, unit='s')).total_seconds()  # Convert back to seconds
        df_selected_larva.index.rename("time", inplace=True)

        myfig.Line(p0, x=df_selected_larva["x"].values[::10], y=df_selected_larva["y"].values[::10], lc=color, zorder=2, lw=0.25, alpha=0.9)
        #myfig.Scatter(p0, x=[df_selected_larva["x"].values[::10][0]], y=[df_selected_larva["y"].values[::10][0]], lc=color, ps=4, pt='o', pc='white', zorder=3, lw=0.2)
        #myfig.Scatter(p0, x=[df_selected_larva["x"].values[::10][-1]], y=[df_selected_larva["y"].values[::10][-1]], lc=color, ps=4, pt='o', pc='white', zorder=3, lw=0.2)

fig.savepdf(root_path / f"figure1_example_data", open_pdf=True)


