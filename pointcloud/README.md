# Applying Context CLuster to Point Cloud Analysis

Our point cloud classification  implementation is based on [pointMLP](https://github.com/ma-xu/pointMLP-pytorch). Thank the authors for their wonderful works.

## Note
Please note that we just simply follow the hyper-parameters of pointMLP which may not be the optimal ones for Context Cluster. 
Feel free to tune the hyper-parameters to get better performance. 


## Usage

Install pointMLP required libs, see [README in pointMLP](https://github.com/ma-xu/pointMLP-pytorch). 


## Data preparation

We don't need to download the ScanObjectNN dataset by ourself. The dataset will be automatically downloaded at the first running. 


## Results and models

| Method | mACC | OA | Download |
| --- | --- | --- | --- |
| PointMLP_CoC | 84.4| 86.2| [log & model](https://drive.google.com/drive/folders/1R5nQTp9mnza3FdqA0FosRj_mzvn4He1F?usp=sharing) |


## Train
To train pointMLP_CoC, run:
```
python main.py --model pointMLP_CoC
```


