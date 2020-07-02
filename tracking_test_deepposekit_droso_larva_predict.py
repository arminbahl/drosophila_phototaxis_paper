from pathlib import Path
import imageio
import numpy as np
from PIL import Image
from deepposekit.models import load_model
import tensorflow as tf
import cv2

print(tf.__version__, tf.config.experimental.list_physical_devices('GPU'))

filename = Path('/n/home10/abahl/engert_storage_armin/maxwell_paper/deepposekit_training/2018_11_15_fish006_setup1.avi')
vid = imageio.get_reader(filename, 'ffmpeg')
input_fps = vid.get_meta_data()['fps']
images = []
for i, frame in enumerate(vid):

    img = np.array(Image.fromarray(frame[:, :, 0]).resize((96, 96)), dtype=np.float32)

    img = 255 * img / img.max()

    #img = (255 * img / img.max()).astype(np.uint8)
    img = cv2.blur(img, (5, 5))
    img[img < 100] = 0


    # img = (255 * img / img.max()).astype(np.uint8)
    #
    # ret, thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #
    # areas = [cv2.contourArea(c) for c in contours]
    # img *= 0
    # if len(areas) > 0:
    #
    #     max_index = np.argmax(areas)
    #     cnt = contours[max_index]
    #
    #     #pl.imshow(thresh1)
    #
    #     cv2.drawContours(img, [cnt], -1, 255, -1)

    images.append(img.astype(np.uint8))
    if i % 10 == 0:
        print(i)
    if i == 25000:
        break

images = np.moveaxis([images], 0, -1).astype(np.uint8)

print(images.shape)
model = load_model(r'/n/home10/abahl/engert_storage_armin/maxwell_paper/deepposekit_training/my_best_model.h5')
predictions = model.predict(images)

np.save('/n/home10/abahl/engert_storage_armin/maxwell_paper/deepposekit_training/my_predictions.npy', predictions)