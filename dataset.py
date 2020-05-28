from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, transform_label, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform
        self.transform_label = transform_label

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            bytes_img = txn.get(key)

            key_label = f'label-{str(index).zfill(5)}'.encode('utf-8')
            bytes_label = txn.get(key_label)

        buffer = BytesIO(bytes_img)
        img = Image.open(buffer)
        img = self.transform(img)

        buffer_label = BytesIO(bytes_label)
        label = Image.open(buffer_label)
        label = self.transform_label(label)

        return img, label
