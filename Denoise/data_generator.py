import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2

patch_size, stride = 40, 10
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128
color = True

class DenoisingDataset(Dataset):

    def __init__(self, src_img, sigma):

        self.sigma = sigma
        self.src_img = src_img
        
    def __getitem__(self, idx):
        batch_x = self.src_img[idx]
        noise = torch.randn(batch_x.shape).mul_(self.sigma/255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.src_img.shape[0]

def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

# get color patches from one color image
def gen_patches(file_name):

    if color:
        img_bgr = cv2.imread(file_name, 3)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = np.asarray(img).shape
    else:
        img = cv2.imread(file_name, 0)
        h, w = np.asarray(img).shape

    crops = 180
    starth = int(h / 2)
    startw = int(w / 2)

    if color:
        img1 = np.zeros([crops, crops, 3])
        img1 = img[int(starth -crops /2) : int(starth + crops /2), int(startw -crops /2):int(startw +crops /2), : ]
        h, w, c = np.asarray(img1).shape
    else:
        img1 = np.zeros([crops, crops])
        img1 = img[int(starth -crops /2) : int(starth + crops /2), int(startw -crops /2):int(startw +crops /2)]
        h, w = np.asarray(img1).shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img1, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
        #plt.imshow(img_scaled)
        #plt.show()
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):

                if color:
                    x = img_scaled[i:i+patch_size, j:j+patch_size, :]
                else:
                    x = img_scaled[i:i+patch_size, j:j+patch_size]
                #print(x.shape)
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                   #print(x_aug.shape,'\n')
                    patches.append(x_aug)
    return patches

def datagenerator(data_dir='/home/x227guo/workspace/MEFNet/color_denoise_dataset/train/CBSD432_1'):
    
    file_list = glob.glob(data_dir+'/*.png') 
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:    
            data.append(patch)
            #print(patch.shape)
    data = np.array(data, dtype='uint8')

    if not color:
        data = np.expand_dims(data, axis=3)
    discard_n = len(data)-len(data)//batch_size*batch_size 
    data = np.delete(data, range(discard_n), axis=0)
    return data

if __name__ == "__main__":
    
    root_src_dir = '/home/x227guo/workspace/MEFNet/color_denoise_dataset/train/CBSD432_1'
    data = datagenerator(data_dir=root_src_dir)
    print(data.shape)

    