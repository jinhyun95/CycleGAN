import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, nparray):
        self.nparray = nparray

    def __len__(self):
        return self.nparray.shape[0]

    def __getitem__(self, idx):
        return self.nparray[idx, :, :, :]

def read_images(path, image_size=64):
    image_names = os.listdir(path)
    images = []
    for name in image_names:
        image = cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (image_size, image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2, 0, 1)
        images.append(image)
    images = np.stack(images)
    return images

def shuffle_array(data):
    index = range(len(data))
    np.random.shuffle(index)
    return data[np.array(index)]

def loader(data, batch_size, shuffle):
    return DataLoader(ImageDataset(data), batch_size=batch_size, shuffle=shuffle)

def data_read(path, domain_name1, domain_name2, image_size, batch_size, separate=False, test_ratio=0.05):
    if separate:
        A = read_images(os.path.join(os.path.join(path, domain_name1), 'training'), image_size)
        A_te = read_images(os.path.join(os.path.join(path, domain_name1), 'test'), image_size)
        B = read_images(os.path.join(os.path.join(path, domain_name2), 'training'), image_size)
        B_te = read_images(os.path.join(os.path.join(path, domain_name2), 'test'), image_size)
        A_tr = shuffle_array(A)
        B_tr = shuffle_array(B)
    else:
        A = read_images(os.path.join(path, domain_name1), image_size)
        B = read_images(os.path.join(path, domain_name2), image_size)
        A_len = A.shape[0]
        B_len = B.shape[0]
        A = shuffle_array(A)
        B = shuffle_array(B)
        A_te = A[:int(A_len*test_ratio), :, :, :]
        A_tr = A[int(A_len*test_ratio):, :, :, :]
        B_te = B[:int(B_len * test_ratio), :, :, :]
        B_tr = B[int(B_len * test_ratio):, :, :, :]
    A_tr = A_tr[:min(A_tr.shape[0], B_tr.shape[0]), :, :, :]
    B_tr = B_tr[:min(A_tr.shape[0], B_tr.shape[0]), :, :, :]
    return loader(A_tr, batch_size, True), loader(A_te, batch_size, False),\
           loader(B_tr, batch_size, True), loader(B_te, batch_size, False)

from scipy.misc import imsave
if __name__ == '__main__':
    loader_arr = data_read('./data', 'horse', 'zebra', 256, 4, True)
    l = 0
    for loader in loader_arr:
        for data in loader:
            a = (data.numpy().transpose(0,2,3,1) * 255.).astype(np.uint8)
            a = [np.squeeze(x) for x in np.split(a, a.shape[0], 0)]
            for i in range(len(a)):
                imsave(os.path.join('./', str(l + 1) + str(i + 1) + '.jpg'), a[i])
            l += 1