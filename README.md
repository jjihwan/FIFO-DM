## Setup

```bash
python3 -m venv .fifodm
source .fifodm/bin/activate
pip3 install -r requirements.txt
```

## Sampling 

You can sample from our **pre-trained Latte models** with [`sample.py`](sample/sample.py). Weights for our pre-trained Latte model can be found [here](https://huggingface.co/maxin-cn/Latte).  The script has various arguments to adjust sampling steps, change the classifier-free guidance scale, etc. For example, to sample from our model on FaceForensics, you can use:

```bash
bash sample/ffs.sh
```

or if you want to sample hundreds of videos, you can use the following script with Pytorch DDP:

```bash
bash sample/ffs_ddp.sh
```

If you want to try generating videos from text, just run `bash sample/t2v.sh`. All related checkpoints will download automatically.

If you would like to measure the quantitative metrics of your generated results, please refer to [here](docs/datasets_evaluation.md).

## Training

We provide a training script for Latte in [`train.py`](train.py). The structure of the datasets can be found [here](docs/datasets_evaluation.md). This script can be used to train class-conditional and unconditional
Latte models. To launch Latte (256x256) training with `N` GPUs on the FaceForensics dataset 
:

```bash
torchrun --nnodes=1 --nproc_per_node=N train.py --config ./configs/ffs/ffs_train.yaml
```


We also provide the video-image joint training scripts [`train_with_img.py`](train_with_img.py). Similar to [`train.py`](train.py) scripts, these scripts can be also used to train class-conditional and unconditional
Latte models. For example, if you want to train the Latte model on the FaceForensics dataset, you can use:

```bash
torchrun --nnodes=1 --nproc_per_node=N train_with_img.py --config ./configs/ffs/ffs_img_train.yaml
```

