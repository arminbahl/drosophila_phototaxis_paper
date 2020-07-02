from pathlib import Path
import numpy as np
from deepposekit.models import load_model
import tensorflow as tf
import imageio
from PIL import Image
import cv2
import sys

print(tf.__version__, tf.config.experimental.list_physical_devices('GPU'))

root_path = Path("/n/home10/abahl/engert_storage_armin/maxwell_paper")

# setup_ID = int(sys.argv[1])

experiment_names = ["virtual_valley_stimulus_control_drosolarva",
                    "virtual_valley_stimulus_drosolarva"]
# if setup_ID == 0:
#     setup_name = "setup0"
# if setup_ID == 1:
#     setup_name = "setup1"

for setup_name in ["setup0", "setup1"]:
    for experiment_name in experiment_names:

        for larva_path in (root_path / setup_name / experiment_name).iterdir():
            larva_ID = f"{larva_path.name}_{setup_name}"

            if "_fish" not in larva_path.name:
                continue

            vid = imageio.get_reader(larva_path / "fish_roi.avi", 'ffmpeg')

            roi_movie = []

            for i, img in enumerate(vid):
                img = np.array(Image.fromarray(img[:, :, 0]).resize((96, 96)), dtype=np.float32)

                img = 255 * img / img.max()
                img = cv2.blur(img, (5, 5)).astype(np.uint8)
                img[img < 100] = 0

                roi_movie.append(img)

            roi_movie = np.moveaxis([roi_movie], 0, -1).astype(np.uint8)

            print("Roi images has the following shape", roi_movie.shape)
            model = load_model(r'/n/home10/abahl/engert_storage_armin/maxwell_paper/deepposekit_training/my_best_model.h5')

            roi_movie_posture = model.predict(roi_movie)

            print("Done prediction, saving....")

            np.save(larva_path / "roi_movie_posture.npy", roi_movie_posture)

