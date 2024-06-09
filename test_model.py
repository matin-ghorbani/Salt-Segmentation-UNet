import os

import torch
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.model import UNet
from utils import config

def log(message, dots=True):
    message = f'[INFO] {message}'
    if dots:
        message += '...'
    print(message)

def prepare_plot(img, original_mask, predicted_mask):
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Image')

    ax[1].imshow(original_mask, cmap='gray')
    ax[1].set_title('Original Mask')

    ax[2].imshow(predicted_mask, cmap='gray')
    ax[2].set_title('Predicted Mask')

    figure.tight_layout()
    # figure.show()
    plt.show()



def predict(model: UNet, img_path: str):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (128, 128))
    original_img = img.copy()

    img = img.astype(np.float32) / 255.

    file_name = img_path.split(os.path.sep)[-1]
    ground_truth_path = os.path.join(config.MASK_DATASET_PATH, file_name)

    original_mask = cv.imread(ground_truth_path, 0)
    original_mask = cv.resize(
        original_mask,
        (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_HEIGHT)
    )

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).to(config.DEVICE)

    predicted_mask = model(img).squeeze()
    predicted_mask = torch.sigmoid(predicted_mask)
    predicted_mask = predicted_mask.cpu().detach().numpy()

    # Filter out the weak predictions and convert them to integers
    predicted_mask = (predicted_mask > config.THRESHOLD) * 255
    predicted_mask = predicted_mask.astype(np.uint8)

    prepare_plot(original_img, original_mask, predicted_mask)


log('Loading up test image paths')
with open(config.TEST_PATHS) as file:
    img_paths = file.read().strip().split('\n')
img_paths = np.random.choice(img_paths, size=10)

log('Loading up the UNet model')
unet: UNet = torch.load(config.MODEL_PATH).to(config.DEVICE)

for path in img_paths:
    predict(unet, path)
