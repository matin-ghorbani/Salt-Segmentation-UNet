import os
from argparse import ArgumentParser, BooleanOptionalAction

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


def predict(model: UNet, img: np.ndarray):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (128, 128))

    img = img.astype(np.float32) / 255.

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).to(config.DEVICE)

    predicted_mask = model(img).squeeze()
    predicted_mask = torch.sigmoid(predicted_mask)
    predicted_mask = predicted_mask.cpu().detach().numpy()

    # Filter out the weak predictions and convert them to integers
    predicted_mask = (predicted_mask > config.THRESHOLD) * 255
    predicted_mask = predicted_mask.astype(np.uint8)

    return predicted_mask


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img', type=str, required=True,
                        help='Your image path to make prediction on it')
    parser.add_argument('--model', type=str, default='UNet_tgs_salt.pth',
                        help='Your model path (default: UNet_tgs_salt.pth)')
    parser.add_argument('--show', type=bool, action=BooleanOptionalAction,
                        default=True, help='Show the predicted mask')
    parser.add_argument('--save', type=bool, action=BooleanOptionalAction,
                        default=True, help='Save the predicted mask')

    opt = parser.parse_args()

    log('Loading up the UNet model')
    unet: UNet = torch.load(opt.model).to(config.DEVICE)

    img = cv.imread(opt.img)
    log('Make predictions')
    prediction = predict(unet, img)

    if opt.show:
        cv.imshow('predictions', prediction)
        cv.waitKey(0)
    
    if opt.save:
        os.makedirs('results', exist_ok=True)
        cv.imwrite('results/result.png', prediction)
        log('Predicted mask saved.', dots=False)
