import pandas as pd
import pickle
from pathlib import Path
import imageio
import numpy as np
from PIL import Image
#from deepposekit.models import load_model
#import tensorflow as tf
import cv2

#print(tf.__version__, tf.config.experimental.list_physical_devices('GPU'))

root_path = Path("/n/home10/abahl/engert_storage_armin/maxwell_paper")

setup_names = ["setup0", "setup1"]
experiment_names = ["virtual_valley_stimulus_control_drosolarva",
                    "virtual_valley_stimulus_drosolarva"]

all_data = dict({"larva_ID": [],
                 "experiment_name": [],
                 "time": [],
                 "x": [],
                 "y": [],
                 "r": [],
                 "roi_movie_framenum": [],
                 "head_x": [],
                 "head_y": [],
                 #"head_node_x": [],
                 #"head_node_y": [],
                 "center_x": [],
                 "center_y": [],
                 #"tail_node_x": [],
                 #"tail_node_y": [],
                 "tail_x": [],
                 "tail_y": []})

for setup_name in setup_names:
    for experiment_name in experiment_names:

        for larva_path in (root_path / setup_name / experiment_name).iterdir():
            larva_ID = f"{larva_path.name}_{setup_name}"

            if "_fish" not in larva_path.name:
                continue

            #if larva_ID != "2018_11_15_fish006_setup1":
            #    continue

            roi_movie_framenum = []

            t0 = 0  # Stich back together the trials
            for trial in range(30):
                print(t0, larva_ID)

                try:
                    f = open(larva_path / "raw_data" / f"trial{trial:03d}.dat", 'rb')
                    data = pickle.load(f)
                    f.close()
                except:
                    print(f"No data for trial {trial}", larva_path)
                    continue

                # Concatencate all data and transform coordinates into cm
                all_data["larva_ID"].extend([larva_ID]*len(data["raw_stimulus_000"]["timestamp"]))
                all_data["experiment_name"].extend([experiment_name] * len(data["raw_stimulus_000"]["timestamp"]))
                all_data["time"].extend(data["raw_stimulus_000"]["timestamp"] + t0)
                all_data["x"].extend(data["raw_stimulus_000"]["fish_position_x"]*6)
                all_data["y"].extend(data["raw_stimulus_000"]["fish_position_y"]*6)
                all_data["r"].extend(np.sqrt(data["raw_stimulus_000"]["fish_position_x"]**2 +
                                             data["raw_stimulus_000"]["fish_position_y"]**2)*6)

                roi_movie_framenum.extend(data["raw_stimulus_000"]["fish_movie_framenum"])

                t0 += data["raw_stimulus_000"]["timestamp"][-1] + 1/90.  # remember the last time and add one delta t

            #if (larva_path / "roi_movie_posture_simple.npy").exists() == False:
            #    print(larva_path / "roi_movie_posture_simple.npy", "does not exist.")
            #    continue

            roi_movie_posture = np.load(larva_path / "roi_movie_posture_simple.npy")

            head_x = roi_movie_posture[:, 0]
            head_y = roi_movie_posture[:, 1]
            #head_node_x = roi_movie_posture[:, 1, 0]
            #head_node_y = roi_movie_posture[:, 1, 1]
            center_x = roi_movie_posture[:, 2]
            center_y = roi_movie_posture[:, 3]
            #tail_node_x = roi_movie_posture[:, 3, 0]
            #tail_node_y = roi_movie_posture[:, 3, 1]
            tail_x = roi_movie_posture[:, 4]
            tail_y = roi_movie_posture[:, 5]

            # Only store the relevant information
            all_data["roi_movie_framenum"].extend(roi_movie_framenum)
            all_data["head_x"].extend(head_x[roi_movie_framenum])
            all_data["head_y"].extend(head_y[roi_movie_framenum])
            #all_data["head_node_x"].extend(head_node_x[roi_movie_framenum])
            #all_data["head_node_y"].extend(head_node_y[roi_movie_framenum])
            all_data["center_x"].extend(center_x[roi_movie_framenum])
            all_data["center_y"].extend(center_y[roi_movie_framenum])
            #all_data["tail_node_x"].extend(tail_node_x[roi_movie_framenum])
            #all_data["tail_node_y"].extend(tail_node_y[roi_movie_framenum])
            all_data["tail_x"].extend(tail_x[roi_movie_framenum])
            all_data["tail_y"].extend(tail_y[roi_movie_framenum])

df = pd.DataFrame.from_dict(all_data)

df.set_index(["larva_ID", "experiment_name", "time"], inplace=True)
df.sort_index(inplace=True)

df.to_hdf(root_path / "all_data.h5", key="raw_data", complevel=4)
