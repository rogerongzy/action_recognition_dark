# action_recognition_dark

## Dataset
video_data_formating.py is used for transfering videos into frames and name them according to a certain rule like './frame/v_action_gvideoname/frame000001.jpg'. Make sure the .py is at the same path as train/validate/test.

```Cmd
python3 video_data_formating.py
```

## Decomposition and Restoration Network 
The decomposition and restoration network is referring to RRDNet at https://github.com/aaaaangel/RRDNet. 

### 1. Prerequisites

* Python 3
* PyTorch >= 0.4.1
* PIL >= 6.1.0
* Opencv-python>=3.4

### 2. Usage
The raw frames pending processing should be placed at the './input' folder.

```Cmd
python3 pipline.py
```


## Super-Resolution Reconstruction
The decomposition and restoration network is referring to Resl-ESRGAN at https://github.com/xinntao/Real-ESRGAN. Installation and usage can be referred to the repository of Real-ESRGAN. The frames pending processing should be placed at the './inputs' folder.


## Conv3D

### 1. Prerequisites

- [Python 3.6](https://www.python.org/)
- [PyTorch 1.0.0](https://pytorch.org/)
- [Numpy 1.15.0](http://www.numpy.org/)
- [Sklearn 0.19.2](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)

### 2. Usage

```Cmd
python3 3DCNN.py
```

The data is put in the folder of './frame' and weights will be saved in './models'. Conv3D_check_prediction.py is used for prediction.
