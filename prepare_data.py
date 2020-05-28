import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_label_file, sizes, resample, mask_size):
    img_file, label_file = img_label_file
    i, img_file = img_file
    _, label_file = label_file

    img = Image.open(img_file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes, resample=resample)

    label = Image.open(label_file)
    label = resize_and_convert(label, mask_size, resample=resample)

    return i, out, label


def prepare(env, dataset, n_worker, sizes=(128, 256, 512, 1024),
            resample=Image.LANCZOS, mask_size=64):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample,
                        mask_size=mask_size)

    files = sorted(dataset.imgs, key=lambda x: x[0])
    files_img = [(i, file) for i, (file, label) in enumerate(files) if 'img'
                 in file]
    files_label = [(i, file) for i, (file, label) in enumerate(files) if 'label'
                   in file]
    files_img_label = list(zip(files_img, files_label))
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, label in tqdm(pool.imap_unordered(resize_fn,
                                                       files_img_label)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            with env.begin(write=True) as txn:
                key = f'label-{str(i).zfill(5)}'.encode('utf-8')
                txn.put(key, label)

            total += 1

        with env.begin(write=True) as txn:
            txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--size', type=str, default='128,256,512,1024')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--resample', type=str, default='lanczos')
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    
    resample_map = {'lanczos': Image.LANCZOS, 'bilinear': Image.BILINEAR}
    resample = resample_map[args.resample]
    
    sizes = [int(s.strip()) for s in args.size.split(',')]

    print(f'Make dataset of image sizes:', ', '.join(str(s) for s in sizes))

    imgset = datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
