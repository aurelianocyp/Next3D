

## Requirements

* 1&ndash;8 high-end NVIDIA GPUs. We have done all testing and development using V100, RTX3090, and A100 GPUs.
* 64-bit Python 3.9 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch install instructions.
* CUDA toolkit 11.3 or later.
* Python libraries: see [environment.yml](./environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `cd Next3D`
  - `conda env create -f environment.yml`
  - `conda activate next3d`
  - `pip install pydantic==1.10.2`
  - pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
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

Download our pretrained models following （next3d_ffhq_512.pkl）[the link](https://drive.google.com/drive/folders/1rbR5ZJ6LQYUSd5J5BkoVYNon_-Lb7KsZ?usp=share_link) and put it under `pretrained_models`. 

For training Next3D on the top of EG3D, please also download the pretrained checkpoint `ffhqrebalanced512-64.pkl` of [EG3D](https://github.com/NVlabs/eg3d/blob/main/docs/models.md).


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

### Installation(这一步最好做一下)

Ensure the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) submodule is properly initialized
```.bash
git submodule update --init --recursive
```
如果中途下载失败了先使用 `git submodule deinit --all`清理一下

Download the pretrained models for FLAME estimation following [DECA](https://github.com/yfeng95/DECA)(deca_model.tar) and put them into `dataset_preprocessing/ffhq/deca/data`; 

download the pretrained models for gaze estimation through the [link](https://drive.google.com/drive/folders/1Jgej9q5W2IYXRa-CWCldyTVXeHk-Oi-I?usp=share_link)(faceverse_v3.npy exBase_52.npy faceverse_v3_old.npy) and put them into `dataset_preprocessing/ffhq/faceverse/data`.

### Preparing datasets（可以不做这一步，不做的话就直接下载网盘里obama-modified.tar放到data文件夹中）

The video reenactment input contains three parts: camera poses `dataset.json`, FLAME meshes ('.obj') and 2D landmark files ('.txt'). For quick start, you can download the processed talking video of President Obama [here](https://drive.google.com/file/d/1ph77uSlLz-xIVlBxwXP3Et7lTR0zHXQR/view?usp=sharing) and place the downloaded folder as `data/obama`. You can also preprocess your custom datasets by running the following commands:

```.bash
cd dataset_preprocessing/ffhq
python preprocess_in_the_wild.py --indir=INPUT_IMAGE_FOLDER
```
preprocessinthewild里是执行了多个python程序,建议一个一个分开运行（通过注释其他行）
建议使用eg3d的环境进行数据处理。

batch_mtcnn.py可以在现有环境做。但是需要：
- `pip install mtcnn`
- `pip uninstall scipy`
- `pip install tensorflow`
- `pip install scipy`
- `git clone https://github.com/NVlabs/nvdiffrast`
- `cd nvdiffrast    # ./Deep3DFaceRecon_pytorch/nvdiffrast`
- `pip install .`

在ffhq文件夹下创建indir的文件夹，将png图片放置在indir文件夹内就行。运行完batch_mtcnn后在indir文件夹内会自动创建detections文件夹，里面放着各自图片对应的txt文件。

建议别在原环境run Deep3DFaceRecon，因为我怀疑现环境并不包括run Deep3DFaceRecon需要的库，最好还是另外配置环境(或者使用eg3d的环境).使用Deep3DFaceRecon_pytorch里的environment吧。test程序完全是d3drf里的，到d3drf里面去配置环境并运行就行。里面的多个python程序分开一步一步运行就行（建议重开一个虚拟机，因为3090太高级了不满足他的环境要求）

crop_images_in_the_wild.py运行的时候，存储图片的文件夹要与程序在同目录，图片都存储在目录下即可，目录下还需要一个detections的文件夹，可以直接将d3dfr里提供的预置数据集拿过来，这个程序的结果也只是在inputdir中创建一个crop文件夹，文件夹中是裁剪之后的图片。如果报prepocess.py 202行的错，那么参考https://blog.csdn.net/m0_53127772/article/details/132492224。无需先运行test程序

3dface2idr_mat.py运行前需要进入deep3dfacerecon目录，然后`git clone https://github.com/deepinsight/insightface.git`,然后在fffhq中cp -r Deep3DFaceRecon_pytorch/insightface/recognition/arcface_torch Deep3DFaceRecon_pytorch/models/。创建Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/<indir>/epoch_20_000000文件夹（注意，都是文件夹）。创建Deep3DFaceRecon_pytorch/BFM/01_MorphableModel.mat Deep3DFaceRecon_pytorch/BFM/Exp_Pca.bin 然后把BFM复制到ffhq目录下（可以先拖到ffhq下然后用命令复制cp -r BFM Deep3DFaceRecon_pytorch/）.结果会在crop中创建camera.json文件。但是需要先进行test操作，也就是得保障Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/<indir>/epoch_20_000000文件夹内有mat，obj，png文件，才能得到正确的camera.json


You will obtain FLAME meshes and 2D landmark files for frames and a 'dataset.json'. Please put all these driving files into a same folder for reenactment later. 


### Reenacting samples
seeds代表source identity.效果比较好： 25,49,85，90, 99, 166，164, 178(小孩),258(小孩),269,287,299(女生)，300（小孩），397（女生）  不太好：，13,89（女生）,165（女）,298,277(女性)，398（女性）
```.bash
python reenact_avatar_next3d.py --drive_root=data/obama --grid=1x1 --seeds=166 --trunc=0.7 --lms_cond=1
```
如果要固定相机参数：--fixed_camera=True

需要使用什么三维模型就将什么三维模型放在obama文件夹中就行，可以改个名，防止与原obama重合.输出在out文件夹中

kpt2d.txt文件必须有

kpt3d.txt可以删除

png文件好像只是起索引的作用，文件名有用，内容没用

obj文件特别有用，控制表情和唇部的

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

## one-shot
https://github.com/MrTornado24/Next3D/issues/27

https://github.com/NVlabs/eg3d/issues/28

https://github.com/MrTornado24/Next3D/issues/38

把pti压缩包下载到tmp中。使用next3d的环境, 参考原生pti的文档。先处理数据，然后训练即可`python scripts/run_pti.py --pivotal_tuning --mesh_path=myphoto.obj --label_path=myphoto.json `。将obj文件和json放在住目录下，data文件中只放png，无需进行pti原始文档里的数据处理，直接运行scripts就行。一些注意事项：

configs中paths_config中的dlib后面的dat路径改为`dlib = './pretrained_models/shape_predictor_68_face_landmarks.dat'`。

align_data中的pre process images中参数填上data。

json文件就是常规的camera.json那种，但是其中必须有图片名+.png的那一个

将configs training criteria utils dnnlib models torch_utils training_avatar_texture文件夹复制到scripts文件夹中

将scripts中的configs中的paths py里的eg3d_ffhq改为eg3d_ffhq = './pretrained_models/v10/network-snapshot-001362.pkl'

将scripts/training/coaches/base coach中53行左右的 if label_path is not None:那一块注释掉。将同一代码中的get_label文件try下第一行改为`label = np.load(self.label_path)`。需要将相应照片的npy文件放在主目录下，npy怎么获得参看eg3d-priojector内容

将get-verts函数中with open改为with open(self.mesh_path) as f:

将scripts/training/projector中的w projector eg3d中的64行的 in G.synthesis.named_buffers()改为 in G.named_buffers()。下载vgg16.pt到主目录，将#url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'注释掉，并将下一行的url改为vgg16.pt。将两处[1,G.mapping.num_ws, 1]改为[1,18, 1].103行 c=label, v=verts

运行完成后，latent code会保存在embeddings目录中，也就是reenact_avatar_texture_fixed_w的w_path参数，pti后的模型会保存在checkpoints中。生成形象的命令：

```bash
python reenact_avatar_texture_fixed_w.py --drive_root obama-modified --network checkpoints/model_eg3d_plus_img00000128.pt --outdir out --exp_cond 0 --fname reenact_texture_inversion.mp4 --w_path=embeddings/eg3d_plus/PTI/img00000128/0.pt
```




