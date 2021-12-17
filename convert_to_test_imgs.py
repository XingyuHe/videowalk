# %%
import torch
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import collections
# %%
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor
def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray
def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    return img
def load_image(img_path):
    # H x W x C => C x H x W
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:,:,::-1]
    img = img.copy()
    return im_to_torch(img)
# %%
DAVIS_PATH = "/mnt/disks/disk1/data/DAVIS"
DAVIS_LBL_IMGS_PATH = os.path.join(DAVIS_PATH, "Annotations/480p/bear")
DAVIS_INPUT_IMGS_PATH = os.path.join(DAVIS_PATH, "JPEGImages/480p/bear")

instance_idx = "00071"

instance_img_path = os.path.join(DAVIS_INPUT_IMGS_PATH, instance_idx + ".jpg")
instance_lbl_path = os.path.join(DAVIS_LBL_IMGS_PATH, instance_idx + ".png")

img = load_image(instance_img_path)
lbl = load_image(instance_lbl_path)

print(collections.Counter(np.array(lbl[lbl != 0])))
# %%
# import own validation data
MOUSE_PATH = "/mnt/disks/disk1/data/mouse_validation_data"
MOUSE_IMGS_PATH = os.path.join(MOUSE_PATH, "barObstacleScaling1")
MOUSE_LBL_PATH = os.path.join(MOUSE_PATH, "CollectedData_.csv")

df_lbl = pd.read_csv(MOUSE_LBL_PATH, header=[1, 2], index_col=0)
# %%
features = df_lbl.columns.levels[0]
coord = df_lbl.columns.levels[1]
radius = 5
color = [255, 0, 0]

for idx, row in df_lbl.iterrows():
    lbl_img_path = os.path.join(MOUSE_PATH, idx)
    lbl_img = cv2.imread(lbl_img_path)
    row = row.dropna()

    for f in features:
        if f in row:
            print(row[f])
            print(type(row[f].x))
            cv2.circle(lbl_img, [int(row[f].x), int(row[f].y)], radius, color, -1)
    plt.imshow(lbl_img)
    break
# %%
