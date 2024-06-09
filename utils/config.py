import os
import torch

DATASET_PATH = os.path.join('dataset', 'train')

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, 'images')
MASK_DATASET_PATH = os.path.join(DATASET_PATH, 'masks')

TEST_SPLIT = .15

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PIN_MEMORY = DEVICE == 'cuda'

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

LR = 1e-3
EPOCHS = 40
BATCH_SIZE = 64

INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128

THRESHOLD = .5

BASE_OUTPUT = 'output'

MODEL_PATH = os.path.join(BASE_OUTPUT, 'UNet_tgs_salt.pth')
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plot.png'])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, 'test_paths.txt'])
