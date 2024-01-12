

## Requirements

* 1&ndash;8 high-end NVIDIA GPUs. We have done all testing and development using V100, RTX3090, and A100 GPUs.
* 64-bit Python 3.9 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.3 or later.
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `cd Next3D`
  - `conda env create -f environment.yml`
  - `conda activate next3d`
  - `pip install pydantic==1.10.2`
  - 需要安装pytorch3d，上面三个步骤做完后先安装pytorch3d。用Ubuntu18.04别用20.04。然后先`conda install -c fvcore -c iopath -c conda-forge fvcore iopath`再`conda install -c bottler nvidiacub`再
```shell
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
```
```shell
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .
```
也许也可能要pip uninstall tqdm termcolor tabulate pyyaml protalocker yacs iopath fvcore一下。但是这次安装最重要的是Ubuntu18.04
## Getting started

Download our pretrained models following [the link](https://drive.google.com/drive/folders/1rbR5ZJ6LQYUSd5J5BkoVYNon_-Lb7KsZ?usp=share_link) and put it under `pretrained_models`. For training Next3D on the top of EG3D, please also download the pretrained checkpoint `ffhqrebalanced512-64.pkl` of [EG3D](https://github.com/NVlabs/eg3d/blob/main/docs/models.md).


## Generating media

```.bash
# Generate videos for the shown cases using pre-trained model

python gen_videos_next3d.py --outdir=out --trunc=0.7 --seeds=10720,12374,13393,17099 --grid=2x2 \
    --network=pretrained_models/next3d_ffhq_512.pkl --obj_path=data/demo/demo.obj \
    --lms_path=data/demo/demo_kpt2d.txt --lms_cond=True
```

```.bash
# Generate images and shapes (as .mrc files) for the shown cases using pre-trained model

python gen_samples_next3d.py --outdir=out --trunc=0.7 --shapes=true --seeds=166 \
    --network=pretrained_models/next3d_ffhq_512.pkl --obj_path=data/demo/demo.obj \
    --lms_path=data/demo/demo_kpt2d.txt --lms_cond=True
```

We visualize our .mrc shape files with [UCSF Chimerax](https://www.cgl.ucsf.edu/chimerax/). Please refer to [EG3D](https://github.com/NVlabs/eg3d) for more detailed instructions.输出在out文件夹中


## Reenacting generative avatars

### Installation

Ensure the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) submodule is properly initialized
```.bash
git submodule update --init --recursive
```
如果中途下载失败了先使用 `git submodule deinit --all`清理一下
Download the pretrained models for FLAME estimation following [DECA](https://github.com/yfeng95/DECA) and put them into `dataset_preprocessing/ffhq/deca/data`; download the pretrained models for gaze estimation through the [link](https://drive.google.com/drive/folders/1Jgej9q5W2IYXRa-CWCldyTVXeHk-Oi-I?usp=share_link) and put them into `dataset_preprocessing/ffhq/faceverse/data`.

### Preparing datasets

The video reenactment input contains three parts: camera poses `dataset.json`, FLAME meshes ('.obj') and 2D landmark files ('.txt'). For quick start, you can download the processed talking video of President Obama [here](https://drive.google.com/file/d/1ph77uSlLz-xIVlBxwXP3Et7lTR0zHXQR/view?usp=sharing) and place the downloaded folder as `data/obama`. You can also preprocess your custom datasets by running the following commands:

```.bash
cd dataset_preprocessing/ffhq
python preprocess_in_the_wild.py --indir=INPUT_IMAGE_FOLDER
```

You will obtain FLAME meshes and 2D landmark files for frames and a 'dataset.json'. Please put all these driving files into a same folder for reenactment later. 


### Reenacting samples
seeds代表source identity，目前试过82 166 165
```.bash
python reenact_avatar_next3d.py --drive_root=data/obama \
  --network=pretrained_models/next3d_ffhq_512.pkl \
  --grid=2x1 --seeds=166 --outdir=out --fname=reenact.mp4 \
  --trunc=0.7 --lms_cond=1
```


## Training


Download and process [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) using the following commands. 
```.bash
cd dataset_preprocessing/ffhq
python runme.py
```
You can perform FLAME and landmarks estimation referring to [preprocess_in_the_wild.py](./dataset_preprocessing/ffhq/preprocess_in_the_wild.py). We will also integrate all the preprocessing steps into a script soon. 
The dataset should be organized as below:
```
    ├── /path/to/dataset
    │   ├── meshes512x512
    │   ├── lms512x512
    │   ├── images512x512
    │   │   ├── 00000
                ├──img00000000.png
    │   │   ├── ...
    │   │   ├── dataset.json
```

You can train new networks using `train_next3d.py`. For example:

```.bash
# Train with FFHQ on the top of EG3D with raw neural rendering resolution=64, using 8 GPUs.
python train_next3d.py --outdir=~/training-runs --cfg=ffhq --data=data/ffhq/images512x512 \
  --rdata data/ffhq/meshes512x512 --gpus=8 --batch=32 --gamma=4 --topology_path=data/demo/head_template.obj \
  --gen_pose_cond=True --gen_exp_cond=True --disc_c_noise=1 --load_lms=True --model_version=next3d \
  --resume pretrained_models/ffhqrebalanced512-64.pkl
```
Note that rendering-conditioned discriminator is not supported currently because obtaining rendering is still time-consuming. We are trying to accelerate this process and the training code will keep updating.
  
## One-shot portrait reenactment and stylization

Code will come soon.

## Citation

```
@inproceedings{sun2023next3d,
  author = {Sun, Jingxiang and Wang, Xuan and Wang, Lizhen and Li, Xiaoyu and Zhang, Yong and Zhang, Hongwen and Liu, Yebin},
  title = {Next3D: Generative Neural Texture Rasterization for 3D-Aware Head Avatars},
  booktitle = {CVPR},
  year = {2023}
}
```

## Acknowledgements

Part of the code is borrowed from [EG3D](https://github.com/NVlabs/eg3d) and [DECA](https://github.com/yfeng95/DECA).