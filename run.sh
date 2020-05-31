python prepare_data.py --out dataset.lmdb --n_worker 8 --size 256,512 \
youtube_512_conditional

python -m torch.distributed.launch --nproc_per_node=2 --master_port=8080 \
train.py --batch 4 --size 512 --wandb --n_sample 25 \
--conditional_architecture enc dataset.lmdb

python -m torch.distributed.launch --nproc_per_node=2 --master_port=8080 \
train.py --batch 4 --size 256 --n_sample 25 --conditional_architecture spade \
dataset.lmdb

sudo docker run --rm -d -v wandb:/vol -p 8100:8080 --name wandb-local wandb/local
wandb login --host=http://0.0.0.0:8100