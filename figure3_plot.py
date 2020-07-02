import pandas as pd
from pathlib import Path
import matplotlib
import my_figure as myfig
import pylab as pl
import numpy as np

root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")

phototaxis_index_grid_search = pd.read_hdf(root_path / "all_data_model_profile1.h5", key="phototaxis_index_grid_search")

fig = myfig.Figure(title="Figure 3")

# Only Rule 1
for rule_i in range(7):
    if rule_i == 0:
        rule_name = "No rules"
        rule1 = False
        rule2 = False
        rule3 = False
        rule4 = False

    if rule_i == 1:
        rule_name = "Only rule 1"
        rule1 = True
        rule2 = False
        rule3 = False
        rule4 = False

    if rule_i == 2:
        rule_name = "Only rule 2"
        rule1 = False
        rule2 = True
        rule3 = False
        rule4 = False

    if rule_i == 3:
        rule_name = "Only rule 3"
        rule1 = False
        rule2 = False
        rule3 = True
        rule4 = False

    if rule_i == 4:
        rule_name = "Only rule 4"
        rule1 = False
        rule2 = False
        rule3 = False
        rule4 = True

    if rule_i == 5:
        rule_name = "Rule 2 and 4"
        rule1 = False
        rule2 = True
        rule3 = False
        rule4 = True

    if rule_i == 6:
        rule_name = "Rule 1, 2, 3, and 4"
        rule1 = True
        rule2 = True
        rule3 = True
        rule4 = True

    df = phototaxis_index_grid_search.query("rule1 == @rule1 and rule2 == @rule2 and rule3 == @rule3 and rule4 == @rule4").droplevel(["rule1", "rule2", "rule3", "rule4"])["phototaxis_index_mean"]

    norm = matplotlib.colors.Normalize(vmin=-40, vmax=40)
    cmap = matplotlib.cm.get_cmap('coolwarm')

    p0 = myfig.Plot(fig, num='a', xpos=1.5 + 2 * rule_i, ypos=22, plot_height=1.5, plot_width=1.5, title=rule_name,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Run length multiplier", xmin=-0.6, xmax=4.6, xticks=[0, 1, 2, 3, 4], xticklabels=["0.25", "0.5", "1", "2", "4"],
                    xticklabels_rotation=45,
                    yl="Turn angle multiplier" if rule_i == 0 else '', ymin=-0.6, ymax=4.6, yticks=[0, 1, 2, 3, 4], yticklabels=["0.25", "0.5", "1", "2", "4"] if rule_i == 0 else ['']*5,
                    yticklabels_rotation=45,
                    zmin=-50, zmax=50, colormap=cmap, show_colormap=True if rule_i == 6 else False, zticks = [-50, -25, 0, 14, 25, 50],
                    zl="Performance index\n(% change relative to control\nfor fraction of time\nspent in the dark ring)")

    for i in range(5):
        for j in range(5):
            print(df.index)
            run_length_multiplier = [0.25, 0.5, 1, 2, 4][i]
            turn_amplitude_multiplier = [0.25, 0.5, 1, 2, 4][j]
            myfig.Scatter(p0, x=[i], y=[j], lc='none', pt='s', pc=cmap(norm(df.loc[run_length_multiplier, turn_amplitude_multiplier])),
                          ps=65, vmin=-40, vmax=40)

fig.savepdf(root_path / f"figure3_gridsearch", open_pdf=True)

sdf
print(df.loc[0.5, 0.5])

asd
pl.plot(df["phototaxis_index_mean"].values)
#m = [df.loc[]]

pl.show()

print(df)
sdf
files = ["rule1_gridsearch", "rule2_gridsearch", "rule3_gridsearch", "rule4_gridsearch", "rule123_gridsearch", "rule1234_gridsearch"]
root_path = Path("/Users/arminbahl/Google Drive/Maxwell paper 2019/JEB_submission/response_to_reviews_1/Revised figure 3 data (1)")


df1 = pd.read_hdf(root_path / f"rule123_gridsearch_sheet2.h5", key="data")
df2 = pd.read_hdf(root_path / f"rule123_gridsearch_sheet3.h5", key="data")
df3 = pd.read_hdf(root_path / f"rule123_gridsearch_sheet4.h5", key="data")

for i in range(15):
    pl.plot(df1.loc[:,i], df2.loc[:,i], alpha=0.5, color='C0')
pl.show()
sdf


fig = myfig.Figure(title="Figure 3")

##########
p0 = myfig.Plot(fig, num='a', xpos=1.5, ypos=22, plot_height=1.25, plot_width=1.5, title="Rule 1",
                    lw=1, pc='white', errorbar_area=False, hlines=[0],
                    xl="", xmin=-0.5, xmax=3.5, xticks=[0, 1, 2, 3], xticklabels=["TAM 0.5", "TAM 1", "TAM 2", "Experiment"], xticklabels_rotation=45,
                    yl="Fraction of time in\nthe dark ring\nrelative to control (%)", ymin=-5, ymax=51, yticks=[0, 25, 50])
df = pd.read_hdf(root_path / f"rule1_gridsearch_sheet1.h5", key="data")
print(df)
myfig.Scatter(p0, x=[0], y=[df.loc[1, 0]], yerr=[(df.loc[1, 2] - df.loc[1, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[1], y=[df.loc[4, 0]], yerr=[(df.loc[4, 2] - df.loc[4, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[2], y=[df.loc[7, 0]], yerr=[(df.loc[7, 2] - df.loc[7, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[3], y=[df.loc[9, 0]], yerr=[(df.loc[9, 2] - df.loc[9, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)

##########
p0 = myfig.Plot(fig, num='', xpos=4, ypos=22, plot_height=1.25, plot_width=1.5, title="Rule 2",
                    lw=1, pc='white', errorbar_area=False, hlines=[0],
                    xl="", xmin=-0.5, xmax=3.5, xticks=[0, 1, 2, 3], xticklabels=["TAM 0.5", "TAM 1", "TAM 2", "Experiment"], xticklabels_rotation=45,
                    yl="", ymin=-5, ymax=51, yticks=[0, 25, 50])

df = pd.read_hdf(root_path / f"rule2_gridsearch_sheet1.h5", key="data")

myfig.Scatter(p0, x=[0], y=[df.loc[1, 0]], yerr=[(df.loc[1, 2] - df.loc[1, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[1], y=[df.loc[4, 0]], yerr=[(df.loc[4, 2] - df.loc[4, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[2], y=[df.loc[7, 0]], yerr=[(df.loc[7, 2] - df.loc[7, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[3], y=[df.loc[9, 0]], yerr=[(df.loc[9, 2] - df.loc[9, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)

##########
p0 = myfig.Plot(fig, num='', xpos=6.5, ypos=22, plot_height=1.25, plot_width=1.5, title="Rule 3",
                    lw=1, pc='white', errorbar_area=False, hlines=[0],
                    xl="", xmin=-0.5, xmax=3.5, xticks=[0, 1, 2, 3], xticklabels=["TFM 0.5", "TFM 1", "TFM 2", "Experiment"], xticklabels_rotation=45,
                    yl="", ymin=-5, ymax=51, yticks=[0, 25, 50])

df = pd.read_hdf(root_path / f"rule3_gridsearch_sheet1.h5", key="data")

myfig.Scatter(p0, x=[0], y=[df.loc[3, 0]], yerr=[(df.loc[3, 2] - df.loc[3, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[1], y=[df.loc[4, 0]], yerr=[(df.loc[4, 2] - df.loc[4, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[2], y=[df.loc[5, 0]], yerr=[(df.loc[5, 2] - df.loc[5, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[3], y=[df.loc[9, 0]], yerr=[(df.loc[9, 2] - df.loc[9, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)

##########
p0 = myfig.Plot(fig, num='', xpos=9, ypos=22, plot_height=1.25, plot_width=1.5, title="Rule 4",
                    lw=1, pc='white', errorbar_area=False, hlines=[0],
                    xl="", xmin=-0.5, xmax=3.5, xticks=[0, 1, 2, 3], xticklabels=["TFM 0.5", "TFM 1", "TFM 2", "Experiment"], xticklabels_rotation=45,
                    yl="", ymin=-5, ymax=51, yticks=[0, 25, 50])

df = pd.read_hdf(root_path / f"rule4_gridsearch_sheet1.h5", key="data")

myfig.Scatter(p0, x=[0], y=[df.loc[3, 0]], yerr=[(df.loc[3, 2] - df.loc[3, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[1], y=[df.loc[4, 0]], yerr=[(df.loc[4, 2] - df.loc[4, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[2], y=[df.loc[5, 0]], yerr=[(df.loc[5, 2] - df.loc[5, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)
myfig.Scatter(p0, x=[3], y=[df.loc[9, 0]], yerr=[(df.loc[9, 2] - df.loc[9, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)


##########
p0 = myfig.Plot(fig, num='', xpos=2, ypos=16, plot_height=1.25, plot_width=5.5, title="Rules 1,2,3",
                    lw=1, pc='white', errorbar_area=False, hlines=[0],
                    xl="", xmin=-0.5, xmax=9.5, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], xticklabels=["TAM 0.5, TFM 0.5",
                                                                                      "TAM 0.5, TFM 1",
                                                                                      "TAM 0.5, TFM 2",

                                                                                      "TAM 1 ,TFM 0.5",
                                                                                      "TAM 1, TFM 1",
                                                                                      "TAM 1, TFM 2",

                                                                                      "TAM 2, TFM 0.5",
                                                                                      "TAM 2, TFM 1",
                                                                                      "TAM 2, TFM 2",
                                                                                      "Experiment"], xticklabels_rotation=45,
                    yl="", ymin=-5, ymax=51, yticks=[0, 25, 50])

df = pd.read_hdf(root_path / f"rule123_gridsearch_sheet1.h5", key="data")
for i in range(10):
    myfig.Scatter(p0, x=[i], y=[df.loc[i, 0]], yerr=[(df.loc[i, 2] - df.loc[i, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)


##########
p0 = myfig.Plot(fig, num='', xpos=2, ypos=12, plot_height=1.25, plot_width=5.5, title="Rules 1,2,3,4",
                    lw=1, pc='white', errorbar_area=False, hlines=[0],
                    xl="", xmin=-0.5, xmax=9.5, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], xticklabels=["TAM 0.5, TFM 0.5",
                                                                                      "TAM 0.5, TFM 1",
                                                                                      "TAM 0.5, TFM 2",

                                                                                      "TAM 1 ,TFM 0.5",
                                                                                      "TAM 1, TFM 1",
                                                                                      "TAM 1, TFM 2",

                                                                                      "TAM 2, TFM 0.5",
                                                                                      "TAM 2, TFM 1",
                                                                                      "TAM 2, TFM 2",
                                                                                      "Experiment"], xticklabels_rotation=45,
                    yl="", ymin=-5, ymax=51, yticks=[0, 25, 50])

df = pd.read_hdf(root_path / f"rule1234_gridsearch_sheet1.h5", key="data")
for i in range(10):
    myfig.Scatter(p0, x=[i], y=[df.loc[i, 0]], yerr=[(df.loc[i, 2] - df.loc[i, 0])], lc=['C0'], pt='o', lw=0.5, ps=3, pc='white', zorder=2)


fig.savepdf(root_path / f"figure3", open_pdf=True)
