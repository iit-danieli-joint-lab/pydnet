# PyDnet

### Update: 
If you are looking Android/iOS implementations of PyDnet, take a look here:
https://github.com/FilippoAleotti/mobilePydnet

This repository contains the source code of pydnet, proposed in the paper "Towards real-time unsupervised monocular depth estimation on CPU", IROS 2018.
If you use this code in your projects, please cite our paper:

```
@inproceedings{pydnet18,
  title     = {Towards real-time unsupervised monocular depth estimation on CPU},
  author    = {Poggi, Matteo and
               Aleotti, Filippo and
               Tosi, Fabio and
               Mattoccia, Stefano},
  booktitle = {IEEE/JRS Conference on Intelligent Robots and Systems (IROS)},
  year = {2018}
}
```
For more details:
[arXiv](https://arxiv.org/abs/1806.11430)

Demo video:
[youtube](https://www.youtube.com/watch?v=Q6ao4Jrulns)

## Requirements
* Install Miniconda from https://github.com/robotology/robotology-superbuild/blob/master/doc/install-miniforge.md then run:
```
conda deactivate
conda create -p ./env 
conda activate ./env
conda install python==3.8
pip install tensorflow=2.5 
conda install opencv matplotlib imageio
conda install -c open3d-admin -c conda-forge open3d
```

### Note

The original project was conceived for `Tensorflow 1.8`, this fork is based on the guide at https://www.tensorflow.org/guide/upgrade to automatically upgrade this project.

## Run pydnet on webcam stream

Just launch in a terminal:

```
python webcam.py --checkpoint_dir ./checkpoint/IROS18/pydnet --resolution [1,2,3]
```

## Run pydnet with a given image to get depthmap and point cloud

Just launch in a terminal:

```
python image2depth.py --width 512 --height 256 --checkpoint_dir ./checkpoint/IROS18/pydnet --resolution 1 --path "/path/to/image"
```

Results are stored into the automatically generated `scene_pcd` folder

## Train pydnet from scratch

### Requirements

* `monodepth (https://github.com/mrharicot/monodepth)` framework by Clément Godard

After you have cloned the monodepth repository, add to it the scripts contained in `training_code` folder from this repository (you have to replace the original `monodepth_model.py` script).
Then you can train pydnet inside monodepth framework.

## Evaluate pydnet on Eigen split

To get results on the Eigen split, just run

```
python experiments.py --datapath PATH_TO_KITTI --filenames PATH_TO_FILELIST --checkpoint_dir checkpoint/IROS18/pydnet --resolution [1,2,3]
```

This script generates `disparity.npy`, that can be evaluated using the evaluation tools by Clément Godard 
