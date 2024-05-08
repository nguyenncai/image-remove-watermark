import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "wm-nowm-v1/train"
VAL_DIR = "wm-nowm-v1/valid"
TEST_DIR = "wm-nowm-v1/test"
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
NUM_WORKERS = 1
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
#LAMBDA_GP = 10
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "checkpoint/disc.pth"
CHECKPOINT_GEN = "checkpoint/gen.pth"

both_transform = A.Compose(
    [
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
       ], additional_targets={"image0": "image"},
)


transform_only_input = A.Compose(
    [
     
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.3),
        #A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        #A.HueSaturationValue(p=0.2),
        #A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2()
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2()
        
    ]
)