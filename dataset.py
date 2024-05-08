import numpy as np
import os
import torch
import config
from PIL import Image
from torch.utils.data import Dataset
# from torchvision.utils import save_image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WaterDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))


        split_half = int(image.shape[1]/2)

     
        input_image = image[:, :split_half, :]
        target_image = image[:, split_half:, :]

    


        #do augment
        augmentations = config.both_transform(
            image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]


        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        
        return input_image,target_image



#import os
#import random
#from torchvision.utils import save_image

# Tạo thư mục test_dataset nếu nó chưa tồn tại
#if not os.path.exists('test_dataset'):
#    os.makedirs('test_dataset')

# Tạo một instance của WaterDataset
#dataset = WaterDataset(root_dir='wm-nowm-v1/train')

# Chọn ngẫu nhiên một số hình ảnh
#num_images_to_save = 100  # Số lượng hình ảnh muốn lưu
#indices = random.sample(range(len(dataset)), num_images_to_save)

#for i, idx in enumerate(indices):
#    input_image, target_image = dataset[idx]

    # Lưu hình ảnh input và target
#    save_image(input_image, f'test_dataset/input_image_{i}.png')
#    save_image(target_image, f'test_dataset/target_image_{i}.png')

