# TF2.0 Training 教程

使用VOC 2012 Dataset进行训练。

**环境准备：创建CPU或GPU虚拟环境,  训练推荐GPU环境, 预测时CPU、GPU均可**

创建Tensorflow CPU虚拟环境

```python
# conda命令创建虚拟环境
conda env create -f conda-cpu.yml
# 激活虚拟环境
conda activate yolov3-tf2-cpu
# 退出虚拟环境
conda deactivate
```

创建Tensorflow GPU虚拟环境

```python
# conda命令创建虚拟环境
conda env create -f conda-gpu.yml
# 激活虚拟环境
conda activate yolov3-tf2-gpu
# 退出虚拟环境
conda deactivate
```

主要依赖库及版本：

```python
python==3.7
pip
matplotlib
opencv-python==4.1.1.26
tensorflow-gpu==2.1.0rc1
lxml
tqdm
```

**以下步骤均在虚拟环境下进行,请先激活虚拟环境**


### 1. 下载数据集

VOC官网 [here](http://host.robots.ox.ac.uk/pascal/VOC/)
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar -O ./data/voc2012_raw.tar
mkdir -p ./data/voc2012_raw
tar -xf ./data/voc2012_raw.tar -C ./data/voc2012_raw
ls ./data/voc2012_raw/VOCdevkit/VOC2012 
```

### 2. 生成TFRecord格式数据集

程序位置 tools/voc2012.py 

```bash
python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split train \
  --output_file ./data/voc2012_train.tfrecord

python tools/voc2012.py \
  --data_dir './data/voc2012_raw/VOCdevkit/VOC2012' \
  --split val \
  --output_file ./data/voc2012_val.tfrecord
```

数据集可视化代码：visualize_dataset.py
```
python tools/visualize_dataset.py --classes=./data/voc2012.names
```

功能：从数据集中随机选取一张图片进行可视化，结果保存为 `output.jpg`

### 3. 训练

#### 3.1 进行迁移训练：使用Darknet预训练权重初始化

先下载yolov3-darknet的预训练权重到./data目录并转化为TF格式

```python
wget https://pjreddie.com/media/files/yolov3.weights -O data/yolov3.weights
python convert.py
```

检查TF格式权重是否有效

```python
python detect.py --image ./data/meme.jpg 
```

开始训练：

```python
python train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer darknet \
	--batch_size 16 \
	--epochs 10 \
	--weights ./checkpoints/yolov3.tf \
	--weights_num_classes 80 
```

原始YOLO是在COCO数据集80分类上训练的，此处我们在VOC数据集20分类上进行迁移训练

#### 3.2 从0开始训练，随机初始化权重参数
此方法效果不是很好

```python
python train.py \
	--dataset ./data/voc2012_train.tfrecord \
	--val_dataset ./data/voc2012_val.tfrecord \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--mode fit --transfer none \
	--batch_size 16 \
	--epochs 10 \
```

#### 3.3 训练结果可视化

使用tensorboard工具可视化log文件

```python
tensorboard --logdir=path_to_your_logs
```

### 4. 预测

 单张图片检测

```python
python detect_singal_picture.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3.tf \
	--image ./data/street.jpg
```

输出结果：output.jpg

随机选取验证集上一张图片检测

```python
python detect_singal_picture.py \
	--classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3.tf \
	--tfrecord ./data/voc2012_val.tfrecord
```

调用本地摄像头实时检测

```python
# webcam
python real_time_detect_video_or_webcam.py \
	--video 0 \
    --classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3.tf 
```

检测本地视频文件

```python
python real_time_detect_video_or_webcam.py \
	--video path_to_file.mp4 
    --classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3.tf \

#保存检测结果
python real_time_detect_video_or_webcam.py \
	--video path_to_file.mp4 \
    --output ./output_path_file.mp4\
    --classes ./data/voc2012.names \
	--num_classes 20 \
	--weights ./checkpoints/yolov3.tf \
```

