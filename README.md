

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
python reenact_avatar_next3d.py --drive_root=data/obama --grid=1x1 --seeds=166 --trunc=0.7 --lms_cond=1 --reload_modules=true
```
如果要固定相机参数：--fixed_camera=True

需要使用什么三维模型就将什么三维模型放在obama文件夹中就行，可以改个名，防止与原obama重合.输出在out文件夹中

kpt2d.txt文件必须有

kpt3d.txt可以删除

png文件好像只是起索引的作用，文件名有用，内容没用

obj文件特别有用，控制表情和唇部的

如果在模型源码中加了print却不输出，可能是因为加载模型时使用了persistence.persistent_class类保存了代码后的模型。可以添加reload_modules参数来进行重新加载模型参数，这样就是使用新的代码了。参看：https://github.com/MrTornado24/Next3D/issues/7 https://github.com/MrTornado24/Next3D/issues/15 发现缺少某个模型时从网盘下载就行，然后就是改一下triplane next3d中的alpha_image与rendering stitch就行，不改的话脸部会有鳞片或者叫broken

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

把pti压缩包下载到tmp中。使用next3d的环境, 参考原生pti的文档。先处理数据，然后训练即可`python scripts/run_pti.py --pivotal_tuning --mesh_path=myphoto.obj --label_path=myphoto.json `。将obj文件和json放在住目录下，data文件中只放一个png，无需进行pti原始文档里的数据处理，直接运行scripts就行。一些注意事项：

configs中paths_config中的dlib后面的dat路径改为`dlib = './pretrained_models/shape_predictor_68_face_landmarks.dat'`。eg3d_ffhq改为`eg3d_ffhq = './pretrained_models/v10/network-snapshot-001362.pkl'`

json文件就是常规的camera.json那种，但是其中必须有图片名+.png的那一个

将configs training criteria utils dnnlib models torch_utils training_avatar_texture文件夹复制到scripts文件夹中: `cp -r configs training criteria utils dnnlib models torch_utils training_avatar_texture scripts/`

将scripts/training/coaches/base coach中205行的`label = [self.labels[f'{idx[3:8]}/'+idx+'.png']]`改为`label = [self.labels[idx+'.png']]`，214行也删掉`self.mesh_path, f'{idx[3:8]}/'+`

run_pti中104行default设置为eg3d_plus

将scripts/training/projector中的w plus projector eg3d中70行url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'注释掉，并将下一行的url改为vgg16.pt。从网盘下载vgg16.pt到主目录

如果需要重载模型，最好先将模型的原码打印出来然后复制到triplane.py中，因为模型原码与triplane中差别比较大，少了个注释中的第五步。感觉应该是triplane_v10中的代码，但是也有一点点小差别，重点是在于blended_planes的定义上

运行完成后，latent code会保存在embeddings目录中，也就是reenact_avatar_texture_fixed_w的w_path参数，pti后的模型会保存在checkpoints中。图片结果会保存在tmp中。如果觉得生成的太差，可以使用obama_modified中的obj文件。生成形象的命令：

```bash
python reenact_avatar_texture_fixed_w.py --drive_root obama-modified --network checkpoints/model_eg3d_plus_img00000128.pt --outdir out --exp_cond 0 --fname reenact_texture_inversion.mp4 --w_path=embeddings/eg3d_plus/PTI/img00000128/0.pt --reload_modules=false
```

报错： 
```
File "/root/autodl-tmp/1/PTI_animatable_eg3d/reenact_avatar_texture_fixed_w.py", line 140, in run_video_animation
    camera_params = label_list[os.path.basename(img_path).replace('png', 'jpg')]
KeyError: '00001.jpg'
```
将jpg改为png就行
```
with open(os.path.join(drive root, "dataset exp.json')，'rb') as f:
FileNotFoundError: 「Errno 2l No such file or directory: 'obama-modified/dataset exp.ison
```
报错语句改为dataset.json就行

如果需要改为1*1的网格，就将reenact程序的变量设置为grid_w = 1 grid_h = 1，imgs变量初始化为[]

如果需要固定视角，需要添加`@click.option('--fixed_camera', type=bool, default=False)`, 在运行时需要加入fixed camera参数。在定义run video animation函数时参数中加上`, fixed_camera`。需要在定义intrinsics的下方添加：
```
    if fixed_camera:
        cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
        conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius,
                                                               device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
# 最后在G.synthesis上方添加`if fixed_camera:   camera_params = conditioning_params`
```

如果需要保存每一帧图像，需要在layout grid的参数中添加save=0，并且在需要生成视频时的layout grid函数调用中设置save=1，在layout grid的return前添加：
```python
    if save:
        save_dir = 'out_image/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 生成图片的路径和文件名
        image_path = os.path.join(save_dir, 'image_{}.png'.format(len(os.listdir(save_dir))))

        # 保存图片
        imageio.imwrite(image_path, img)
```

## bug
### 微调时的reload modules
①如果出现了misc162行的assertion error，把那一行注释掉即可，这个问题会发生在reload微调后的模型的时候。②但是根本原因是不重新加载的话，新模型中有两个参数是用torch.nn.Parameter定义的，属于模型内容，而原模型是用torch.tensor定义的，不属于模型内容，导致给新模型赋值时读取不到原模型的值。最好的解决方法不是注释掉assert，而是将triplane中模型的的orth_shift与orth_scale参数改为torch.tensor定义，并且在reload中将老模型的值赋给新模型。③但是，即使能够重新加载模型，生成的质量依旧超级差。建议通过print(G._orig_module_src)输出原码，可以发现原码和triplane中原码非常不一样，干脆将源码复制到triplane中生成吧
### 微调一个多G的那个模型
①先把微调的模型路径改为next3d模型。然后遇到什么bug改什么bug。②可能遇到牙齿那里生成的bug。那是因为vertex没有将kpt2d包含进来。改一下base coach里加载vertex的函数就行。③训练好模型之后生成用真正的生成代码去生成就行。但是又要改bug。可能network stylegan2 styleunet有报错，应该是微调模型中的unet与原始unet不太一样，把原始unet复制过去就行。
### 流程
①从网盘下载next.zip和PTI.zip②在PTI中进行微调，在next中进行生成③微调流程：`python scripts/run_pti.py --pivotal_tuning --mesh_path=myphoto.obj --label_path=myphoto.json`，将checkpoints文件夹里生成的701338997大小（668M）的模型移动到next3d下，将0.pt移动到next3d下④生成流程：将reenact_avatar_next3d1.py中network参数默认值改为刚才移动的模型的名字，然后运行`python reenact_avatar_next3d1.py --drive_root=data/obama --grid=1x1 --seeds=166 --trunc=0.7 --lms_cond=1 --reload_modules=true`



