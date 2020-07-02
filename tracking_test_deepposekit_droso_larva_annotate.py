from deepposekit import Annotator
from deepposekit.io import initialize_dataset
from pathlib import Path
import imageio
import numpy as np
from PIL import Image
import cv2

root_path = Path("/Users/arminbahl/Desktop/preprocessed data/maxwell_paper")
filename = Path(root_path / "2018_11_15_fish006_setup1.avi")
vid = imageio.get_reader(filename, 'ffmpeg')
#
# filename = Path('/Users/arminbahl/Desktop/droso_larva/full_larva_movie.avi')
# vid = imageio.get_reader(filename, 'ffmpeg')
# input_fps = vid.get_meta_data()['fps']

images = []
for i, frame in enumerate(vid):

    if i%5 != 0:
        continue

    if i < 300:
        continue

    img = np.array(Image.fromarray(frame[:, :, 0]).resize((96, 96)), dtype=np.float32)
    img = 255 * img / img.max()

    #img = (255 * img / img.max()).astype(np.uint8)
    img = cv2.blur(img, (5, 5))
    img[img < 100] = 0

    # ret, thresh1 = cv2.threshold(img, 0.5*img.max(), 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # areas = [cv2.contourArea(c) for c in contours]
    # img *= 0
    # if len(areas) > 0:
    #     cnt = contours[np.argmax(areas)]
    #
    #     # pl.imshow(thresh1)
    #
    #     cv2.drawContours(img, [cnt], -1, 255, -1)

    images.append(img.astype(np.uint8))

    if i == 800:
        break

images = np.moveaxis([images], 0, -1).astype(np.uint8)

# initialize_dataset(
#      images=images,
#      datapath=str(root_path / 'my_annotations.h5'),
#      skeleton=str(root_path / 'my_skeleton.csv'),
#      overwrite=True  # This overwrites the existing datapath
# )

app = Annotator(datapath=str(root_path / 'my_annotations.h5'),
                dataset='images',
                skeleton=str(root_path / 'my_skeleton.csv'),
                shuffle_colors=False,
                text_scale=0.1)

app.run()
