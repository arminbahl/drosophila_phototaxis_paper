from pathlib import Path
import imageio
import numpy as np
from PIL import Image
import cv2
from deepposekit.io import DataGenerator
import pylab as pl
import tqdm
import pandas as pd
from scipy.signal import find_peaks
from matplotlib.colors import to_rgb

# filename = Path('/Users/arminbahl/Desktop/preprocessed data/maxwell_paper/2018_11_15_fish006_setup1.avi')
# vid = imageio.get_reader(filename, 'ffmpeg')
# input_fps = vid.get_meta_data()['fps']
# #
# images = []
# for i, frame in enumerate(vid):
#
#     img = np.array(Image.fromarray(frame[:, :, 0]).resize((96, 96)), dtype=np.float32)
#
#     img = 255 * img / img.max()
#
#     #img = (255 * img / img.max()).astype(np.uint8)
#     img = cv2.blur(img, (5, 5))
#     img[img < 100] = 0
#     #
#     # img = (255 * img / img.max()).astype(np.uint8)
#     #
#     # ret, thresh1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
#     # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     #
#     # areas = [cv2.contourArea(c) for c in contours]
#     # img *= 0
#     # if len(areas) > 0:
#     #
#     #     max_index = np.argmax(areas)
#     #     cnt = contours[max_index]
#     #
#     #     #pl.imshow(thresh1)
#     #
#     #     cv2.drawContours(img, [cnt], -1, 255, -1)
#
#     images.append(img.astype(np.uint8))
#     if i % 10 == 0:
#         print(i)
#     if i == 25000:
#         break
#
# images = np.moveaxis([images], 0, -1).astype(np.uint8)
#
# print(images.shape)
#

#data_generator = DataGenerator('/Users/arminbahl/Desktop/preprocessed data/maxwell_paper/my_annotations.h5', mode="full")

roi_movie_posture = np.load('/Users/arminbahl/Desktop/roi_movie_posture_simple.npy')

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

    # dotProduct2 = dx3 * dx4 + dy3 * dy4
    # modOfVectors2 = np.sqrt(dx3**2 + dy3**2) * np.sqrt(dx4**2 + dy4**2)
    #
    # dotProduct3 = dx5 * dx6 + dy5 * dy6
    # modOfVectors3 = np.sqrt(dx5**2 + dy5**2) * np.sqrt(dx6**2 + dy6**2)

    return np.degrees(np.arccos(dotProduct1/modOfVectors1))# + \
           #np.degrees(np.arccos(dotProduct2/modOfVectors2)) + \
           #np.degrees(np.arccos(dotProduct3/modOfVectors3))

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

ang = curvature(head_x, head_y,
                #head_node_x, head_node_y,
                center_x, center_y,
                #tail_node_x, tail_node_y,
                tail_x, tail_y)

df = pd.DataFrame(ang)
df = df.fillna(method='bfill')

#for i in range(1,len(ang)):
#    if np.isnan(ang[i]):
#        ang[i] = ang[i-1]
from my_general_helpers import butter_lowpass_filter

pl.plot(df)
# for i in range(2, len(ang)):
#     if np.abs(ang[i] - ang[i-1] - ang[i-1] + ang[i-2]) > 50:
#         ang[i] = ang[i - 2]
#         ang[i - 1] = ang[i - 2]
#
#     # if np.abs(ang[i] - ang[i-1]) > 50:
#     #     ang[i] = ang[i - 1]
#     #     #ang[i - 1] = ang[i - 2]

#ang = np.nan_to_num(ang)
df["ang_filtered"] = butter_lowpass_filter(df[0], cutoff=3, fs=90., order=5)
pl.plot(df["ang_filtered"])

peaks, _ = find_peaks(df["ang_filtered"], height=40, distance=3 * 90, width=0.5 * 90, prominence=0.5)
peaks = peaks[df["ang_filtered"].iloc[peaks] < 300]

pl.plot(peaks, df["ang_filtered"].iloc[peaks], 'o')
pl.show()
jj
predictions= roi_movie_posture
#predictions = predictions - predictions[:, [1], :]
#
# x1 = predictions[:, 0, 0]
# y1 = predictions[:, 0, 1]
#
# x2 = predictions[:, 1, 0]
# y2 = predictions[:, 1, 1]
#
# x3 = predictions[:, 2, 0]
# y3 = predictions[:, 2, 1]
#
# for j in range(1, predictions.shape[0]):
#
#     # When head and tail are flipped (either they are too close in the same frame, or in concective frames)
#     if np.sqrt((x1[j] - x3[j] - x2[j] + x2[j])**2 + (y1[j] - y3[j] - y2[j] + y2[j])**2) < 10:
#         x1[j] = x1[j - 1]
#         y1[j] = y1[j - 1]
#         x2[j] = x2[j - 1]
#         y2[j] = y2[j - 1]
#         x3[j] = x3[j - 1]
#         y3[j] = y3[j - 1]
#     elif np.sqrt((x1[j] - x1[j-1] - x2[j] + x2[j-1])**2 + (y1[j] - y1[j-1] - y2[j] + y2[j-1])**2) > 10 and \
#             np.sqrt((x3[j] - x3[j-1] - x2[j] + x2[j-1])**2 + (y3[j] - y3[j-1] - y2[j] + y2[j-1])**2) > 10:
#         k_x = x1[j]
#         k_y = y1[j]
#
#         x1[j] = x3[j]
#         y1[j] = y3[j]
#         x3[j] = k_x
#         y3[j] = k_y
#
#
print(predictions.shape)

predictions = predictions[..., :2]
predictions *= 2
#print(predictions)

resized_shape = (data_generator.image_shape[0] * 2, data_generator.image_shape[1] * 2)
#cmap = pl.cm.tab10(np.linspace(0, 1, data_generator.keypoints_shape[0]))[:, :3][:, ::-1] * 255

writer = imageio.get_writer('/Users/arminbahl/Desktop/preprocessed data/maxwell_paper/my_predictions.mp4', codec='libx264', fps=30, ffmpeg_params=["-b:v", "8M"])

node_colors = ["C0", "C1", "C2", "C3", "C4"]
counter = 0
for frame, keypoints in tqdm.tqdm(zip(images, predictions)):

     frame = frame[:,:,0]
     frame = frame.copy()
     frame = cv2.resize(frame, resized_shape)

     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
     cv2.putText(frame, f"{counter}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255))
     for idx, node in enumerate(data_generator.graph):
         if node >= 0:
             pt1 = keypoints[idx]
             pt2 = keypoints[node]
             cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 0, 255), 1, cv2.LINE_AA)
     for idx, keypoint in enumerate(keypoints):
         keypoint = keypoint.astype(int)

         cv2.circle(frame, (keypoint[0], keypoint[1]), 2, tuple(np.array(to_rgb(node_colors[idx]))*255), -1, lineType=cv2.LINE_AA)

     writer.append_data(frame)
     counter+=1

writer.close()
