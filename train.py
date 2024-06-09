import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
from tqdm import tqdm

from utils.model import UNet
from utils.dataset import SegmentationDataset
from utils import config


def log(message, dots=True):
    message = f'[INFO] {message}'
    if dots:
        message += '...'
    print(message)


image_paths = sorted(list(
    paths.list_images(config.IMAGE_DATASET_PATH)
))
mask_paths = sorted(list(
    paths.list_images(config.MASK_DATASET_PATH)
))

train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    image_paths, mask_paths, test_size=config.TEST_SPLIT, random_state=42
)

log('Saving testing image paths')
with open(config.TEST_PATHS, 'w') as file:
    file.write('\n'.join(test_imgs))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)),
    transforms.ToTensor()
])

train_dataset = SegmentationDataset(
    img_paths=train_imgs,
    mask_paths=train_masks,
    transforms=transform
)
test_dataset = SegmentationDataset(
    img_paths=test_imgs,
    mask_paths=test_masks,
    transforms=transform
)

log(f'Found {len(train_dataset)} examples in the training set')
log(f'Found {len(test_dataset)} examples in the testing set')

train_loader = DataLoader(
    train_dataset,
    config.BATCH_SIZE,
    shuffle=True,
    pin_memory=config.PIN_MEMORY,
    num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test_dataset,
    config.BATCH_SIZE,
    pin_memory=config.PIN_MEMORY,
    num_workers=os.cpu_count()
)

unet = UNet().to(config.DEVICE)
optimizer = torch.optim.Adam(unet.parameters(), config.LR)
loss_fn = nn.BCEWithLogitsLoss()

train_steps = len(train_dataset) // config.BATCH_SIZE
test_steps = len(test_dataset) // config.BATCH_SIZE

history = {
    'train_loss': [],
    'test_loss': []
}

log('Training the network')
start_time = time.time()
for epoch in range(1, config.EPOCHS + 1):
    unet.train()

    total_train_loss = 0
    total_test_loss = 0

    for batch in train_loader:
        x, y = map(lambda x: x.to(config.DEVICE), batch)

        y_pred = unet(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss

    with torch.no_grad():
        unet.eval()

        for batch in test_loader:
            x, y = map(lambda x: x.to(config.DEVICE), batch)

            y_pred = unet(x)
            total_test_loss += loss_fn(y_pred, y)

    avg_train_loss = total_train_loss / train_steps
    avg_test_loss = total_test_loss / test_steps

    history['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    history['test_loss'].append(avg_test_loss.cpu().detach().numpy())

    log(f'Epoch {epoch}/{config.EPOCHS}')
    log(f'Train Loss: {avg_train_loss:.4f}')
    log(f'Test Loss: {avg_test_loss:.4f}')

end_time = time.time()
log(f'Training completed in {end_time - start_time:.2f} seconds', dots=False)

plt.style.use('ggplot')
plt.figure()
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['test_loss'], label='test_loss')
plt.title('Training Loss on Dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend(loc='lower left')
plt.savefig(config.PLOT_PATH)

log(f'Saving the model to: {config.MODEL_PATH}')
torch.save(unet, config.MODEL_PATH)
