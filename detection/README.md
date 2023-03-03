# Applying Context Cluster to Object Detection

Our detection implementation is based on [MMDetection](https://github.com/open-mmlab/mmdetection/) and [PVT detection](https://github.com/whai362/PVT/tree/v2/detection). Thank the authors for their wonderful works.


## Note
Please note that we just simply follow the hyper-parameters of PVT which may not be the optimal ones for Context Cluster. 
Feel free to tune the hyper-parameters to get better performance. 



## Usage

Install [MMDetection](https://github.com/open-mmlab/mmdetection/) from souce cocde,

or

```
pip install mmdet --user
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection](https://github.com/open-mmlab/mmdetection/).


## Results and models on COCO

|Backbone|Parmas|AP-box|AP-box@50|AP-box@75|AP-mask|AP-mask@50|AP-mask@75|Download|
|-------------------|--------|--------|-----------|-----------|---------|------------|------------|----------|
|ResNet18|31.2M|34.0|54.0|36.7|31.2|51.0|32.7||
|PVT-Tiny|32.9M|36.7|59.2|39.3|35.1|56.7|37.3||
|**CoC-small-4**|33.6M|35.9|58.3|38.3|33.8|55.3|35.8|[[model]](https://drive.google.com/drive/folders/1-TthSC4bWdKhyc_MW9Qsax9WohrniZWB?usp=sharing)|
|**CoC-small-25**|33.6M|37.5|60.1|40.0|35.4|57.1|37.9|[[model]](https://drive.google.com/drive/folders/1Y-A-GinWSyhl8DWGu5qdJC36qc1FvSnD?usp=sharing)|
|**CoC-small-49**|33.6M|37.2|59.8|39.7|34.9|56.7|37.0|[[model]](https://drive.google.com/drive/folders/1hDFZpy1y8GwN0ZvpoTw8OtPuYvvvw6Fy?usp=sharing)|
|----|----|----|----|----|----|----|----|----|
|ResNet50|44.2M|38.0|58.6|41.4|34.4|55.1|36.7||
|PVT-Small|44.1M|40.4|62.9|43.8|37.8|60.1|40.3||
|**CoC-medium-4**|42.1M|38.6|61.1|41.5|36.1|58.2|38.0|[[model]](https://drive.google.com/drive/folders/1a8X1Qw4z_hABe8jhOAmG6-qM6W0t-GPT?usp=sharing)|
|**CoC-medium-25**|42.1M|40.1|62.8|43.6|37.4|59.9|40.0|[[model]](https://drive.google.com/drive/folders/1kvTZ9EX0rT_XQKVQUeS5536uHCDHh-Bg?usp=sharing)|
|**CoC-medium-49**|42.1M|40.6|63.3|43.9|37.6|60.1|39.9|[[model]](https://drive.google.com/drive/folders/1xCc148wuHq4Y_zxhlzU1ZLw7ZxTp4rdh?usp=sharing)|
## Evaluation

To evaluate Context Cluster + Mask R-CNN on COCO val2017, run:
```
dist_test.sh configs/{configure-file} /path/to/checkpoint_file 8 --out results.pkl --eval bbox segm
```


## Training

To train Context Cluster + Mask R-CNN on COCO train2017:
```
dist_train.sh configs/{configure-file} 8
```
