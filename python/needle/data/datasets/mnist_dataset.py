from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.transforms = transforms

        with gzip.open(image_filename, 'rb') as f:
            data = f.read()
            magic, num_images, rows, cols = struct.unpack_from('>IIII', data, 0)
            assert magic == 2051, "Invalid magic number in image file: {}".format(magic)
            X = np.frombuffer(data, dtype=np.uint8, offset=16)
            X = X.reshape(num_images, rows * cols)
            self.X = X.astype(np.float32) / np.float32(255.0)

        with gzip.open(label_filename, 'rb') as f:
            data = f.read()
            magic, num_labels = struct.unpack_from('>II', data, 0)
            assert magic == 2049, "Invalid magic number in label file: {}".format(magic)
            self.y = np.frombuffer(data, dtype=np.uint8, offset=8)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.X[index]
        labels = self.y[index]
        if self.transforms is None or len(self.transforms) == 0:
            return imgs, labels
        
        is_single_image = len(imgs.shape) == 1
        if is_single_image:
            imgs = [imgs]
        tranformed_imgs = []
        for img in imgs:
            img_3d = img.reshape(28, 28, 1)
            for t in self.transforms:
                img_3d = t(img_3d)
            tranformed_imgs.append(img_3d.reshape(-1))
        
        if is_single_image:
            return tranformed_imgs[0], labels
        else:
            return np.vstack(tranformed_imgs), labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION