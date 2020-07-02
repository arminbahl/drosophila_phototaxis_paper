import pandas as pd
from pathlib import Path
import pylab as pl
import my_figure as myfig
from scipy.stats import ttest_ind, ttest_1samp
import numpy as np
from tqdm import tqdm
import cv2
from deepposekit.io import DataGenerator
from matplotlib.colors import to_rgb
import imageio
from PIL import Image
from my_general_helpers import butter_lowpass_filter

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



def angle_between_points(x1, y1, x2, y2, x3, y3):
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

def luminance_equation(x, y):

    r = np.sqrt(x**2 + y**2)

    return (r - 3) ** 2 / 9


root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")
df_raw_data = pd.read_hdf(root_path / "all_data_deepposekit.h5", key="raw_data")
df_event_data = pd.read_hdf(root_path / "all_data_deepposekit.h5", key="event_data")

df_raw_data["curvature"] = curvature(df_raw_data["head_x"].values,
                            df_raw_data["head_y"].values,
                            #df["head_node_x"].values,
                            #df["head_node_y"].values,
                            df_raw_data["center_x"].values,
                            df_raw_data["center_y"].values,
                            #df["tail_node_x"].values,
                            #df["tail_node_y"].values,
                            df_raw_data["tail_x"].values,
                            df_raw_data["tail_y"].values)

# df_raw_data = df_raw_data.rolling(270, center=True).mean()
#
# df_raw_data["curvature"] = curvature(df_raw_data.shift(periods=90)["x"].values,
#                                          df_raw_data.shift(periods=90)["y"].values,
#                                          df_raw_data.shift(periods=0)["x"].values,
#                                          df_raw_data.shift(periods=0)["y"].values,
#                                          df_raw_data.shift(periods=-90)["x"].values,
#                                          df_raw_data.shift(periods=-90)["y"].values)
#
# df_raw_data["curvature"] = curvature(df_raw_data["head_x"].values,
#                                        df_raw_data["head_y"].values,
#                                        df_raw_data["center_x"].values,
#                                        df_raw_data["center_y"].values,
#                                        df_raw_data["tail_x"].values,
#                                        df_raw_data["tail_y"].values)

#df_raw_data["curvature"] = df_raw_data["curvature"].fillna(method='bfill')

# Filter out all jumps and lost larvae
#df_raw_data["curvature_filtered"] = butter_lowpass_filter(df_raw_data["curvature"], cutoff=3, fs=90., order=5)
# pl.plot(df_raw_data["curvature_filtered"].values)
# pl.show()

df_raw_data_selected_larva = df_raw_data.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and larva_ID == '2018_11_15_fish006_setup1'").reset_index(level=['experiment_name', 'larva_ID'], drop=True)
df_event_data_selected_larva = df_event_data.query("experiment_name == 'virtual_valley_stimulus_drosolarva' and larva_ID == '2018_11_15_fish006_setup1'").reset_index(level=['experiment_name', 'larva_ID'], drop=True)

# Make a movie
# filename = Path(root_path / "2018_11_15_fish006_setup1.avi")
# vid = imageio.get_reader(filename, 'ffmpeg')
#
# writer = imageio.get_writer(root_path / '2018_11_15_fish006_setup1_pose_manual.mp4', codec='libx264', fps=90, ffmpeg_params=["-b:v", "8M"])
#
# node_colors = ["C0", "C1", "C2"]
#
# for t, row in tqdm(df_raw_data_selected_larva.query("time > 930 and time < 1040").iterrows()):
#
#     img = vid.get_data(int(row["roi_movie_framenum"]))[:, :, 0]
#     img = np.array(Image.fromarray(img).resize((96*4, 96*4)), dtype=np.float32)
#     img = (255 * img / img.max()).astype(np.uint8)
#
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
#     cv2.putText(img, f"{t-930:.1f} s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255))
#
#     cv2.line(img, (int(row["head_x"]*4), int(row["head_y"]*4)), (int(row["center_x"]*4), int(row["center_y"]*4)), (0, 0, 255), 2, cv2.LINE_AA)
#     cv2.line(img, (int(row["center_x"]*4), int(row["center_y"]*4)), (int(row["tail_x"]*4), int(row["tail_y"]*4)), (0, 255, 0), 2, cv2.LINE_AA)
#
#     cv2.circle(img, (int(row["head_x"]*4), int(row["head_y"]*4)), 4, tuple(np.array(to_rgb("C0")) * 255), -1, lineType=cv2.LINE_AA)
#     cv2.circle(img, (int(row["center_x"]*4), int(row["center_y"]*4)), 4, tuple(np.array(to_rgb("C1")) * 255), -1, lineType=cv2.LINE_AA)
#     cv2.circle(img, (int(row["tail_x"]*4), int(row["tail_y"]*4)), 4, tuple(np.array(to_rgb("C2")) * 255), -1, lineType=cv2.LINE_AA)
#
#     writer.append_data(img)
#
# writer.close()


fig = myfig.Figure(title="Figure 2")

##########
p0 = myfig.Plot(fig, num='c', xpos=3.5, ypos=22, plot_height=1.25, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="Time (s)", xmin=-5, xmax=105, xticks=[0, 25, 50, 75, 100], hlines=[30],
                    yl="Larval curvature (deg)", ymin=-1, ymax=60.1, yticks=[0, 30, 60])

myfig.Line(p0, x=df_raw_data_selected_larva.query("time > 930 and time < 1030").index - 930,
           y=df_raw_data_selected_larva.query("time > 930 and time < 1030")["curvature"],
           lc='blue', lw=0.5, zorder=1, alpha=0.5)
myfig.Scatter(p0, x=df_event_data_selected_larva.query("time_at_current_turn_event > 930 and time_at_current_turn_event < 1030")["time_at_current_turn_event"] - 930,
              y=df_event_data_selected_larva.query("time_at_current_turn_event > 930 and time_at_current_turn_event < 1030")["curvature_at_current_turn_event"],
              lc='C1', pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)

##########
p0 = myfig.Plot(fig, num='a', xpos=6.5, ypos=22, plot_height=1.25, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="X (cm)", xmin=-4.1, xmax=-4.1+3, xticks=[-4, -3],
                    yl="Y (cm)", ymin=2.8, ymax=2.8+3, yticks=[3, 4])

myfig.Line(p0, x=df_raw_data_selected_larva.query("time > 930 and time < 1030")["x"],
           y=df_raw_data_selected_larva.query("time > 930 and time < 1030")["y"],
           lc='blue', lw=0.5, zorder=1, alpha=0.5)
myfig.Scatter(p0, x=df_event_data_selected_larva.query("time_at_current_turn_event > 930 and time_at_current_turn_event < 1030")["x_at_current_turn_event"],
              y=df_event_data_selected_larva.query("time_at_current_turn_event > 930 and time_at_current_turn_event < 1030")["y_at_current_turn_event"],
              lc='C1', pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)



##########
p0 = myfig.Plot(fig, num='a', xpos=9.5, ypos=22, plot_height=1.25, plot_width=1.5,
                    lw=1, pc='white', errorbar_area=False,
                    xl="X (cm)", xmin=-6.1, xmax=6.1, xticks=[-6, -3, 0, 3, 6],
                    yl="Y (cm)", ymin=-6.1, ymax=6.1, yticks=[-6, -3, 0, 3, 6])


myfig.Line(p0, x=df_raw_data_selected_larva.x, y=df_raw_data_selected_larva.y, lc='blue', lw=0.5, zorder=1, alpha=0.5)
myfig.Scatter(p0, x=df_event_data_selected_larva["x_at_current_turn_event"],
              y=df_event_data_selected_larva["y_at_current_turn_event"],
              lc='C1', pt='o', lw=0.5, ps=2, pc='white', zorder=2, alpha=0.5)

fig.savepdf(root_path / f"figure2_example_data", open_pdf=True)

