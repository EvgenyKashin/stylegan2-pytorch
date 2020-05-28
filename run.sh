python prepare_data.py --out dataset.lmdb --n_worker 8 --size 256 \
youtube_512_conditional

python -m torch.distributed.launch --nproc_per_node=2 --master_port=8080 \
train.py --batch 8 dataset.lmdb