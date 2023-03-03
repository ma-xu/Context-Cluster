# Applying Context Cluster to Semantic Segmentation

Our semantic segmentation implementation is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [PVT segmentation](https://github.com/whai362/PVT/tree/v2/segmentation). Thank the authors for their wonderful works.

## Note
Please note that we just simply follow the hyper-parameters of PVT which may not be the optimal ones for Context Cluster. 
Feel free to tune the hyper-parameters to get better performance. 


## Usage

Install MMSegmentation v0.19.0. 


## Data preparation

Prepare ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


## Results and models

| Method | Backbone | mIoU |Download |
| --- | --- |--- | --- |
| Semantic FPN | CoC-small-4  |     36.6    | [model](https://drive.google.com/drive/folders/1p3QF6shTC-QCHt9Ladj5OjI7D25rso33?usp=sharing) |
| Semantic FPN | CoC-small-25 |    36.4    | [model](https://drive.google.com/drive/folders/1NXC7NjqVeGoF7Y1IDgPj-Hk64VPPF67K?usp=sharing) |
| Semantic FPN | CoC-small-49 |    36.3     | [model](https://drive.google.com/drive/folders/1V7AoSElympAgNdyASZ6VfghUEEaxwQMF?usp=sharing) |
| Semantic FPN | CoC-medium-4 |     40.2    | [model](https://drive.google.com/drive/folders/1CQAXrGoVVot5fcIJDEpx_VDqQaOpyd7u?usp=sharing) |
| Semantic FPN | CoC-medium-25 |     40.6     | [model](https://drive.google.com/drive/folders/1tKYCQSqoOEPtHYBUEAEhzheObt6zBTwZ?usp=sharing) |
| Semantic FPN | CoC-medium-49 |     40.8     | [model](https://drive.google.com/drive/folders/1VLfS9HH8WOM-VYFJo7oiTRut6JjxMuE9?usp=sharing) |

## Evaluation
To evaluate Context Cluster + Semantic FPN on a single node with 4 GPUs run:
```
dist_test.sh configs/sem_fpn/{configure-file} /path/to/checkpoint_file 4 --out results.pkl --eval mIoU
```


## Training
To train Context Cluster + Semantic FPN on a single node with 8 GPUs run:

```
dist_train.sh configs/sem_fpn/{configure-file} 8
```
