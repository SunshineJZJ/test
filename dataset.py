import tensorflow as tf
from absl.flags import FLAGS

# 函数装饰器
@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: [batch_size, yolo_max_boxes, (xmin, ymin, xmax, ymax, label, best_anchor)]
    # anchor_idxs: one of anchor_masks
    
    # batch_size
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [xmin, ymin, xmax, ymax, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    # 数据类型转换为int32
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    # 用于生成y_true_out
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    # N:batch_size大小 遍历每一张图
    for i in tf.range(N):
        # 遍历每一个Bbox
        for j in tf.range(tf.shape(y_true)[1]):
            # 通过xmax判断Bbox是否有效
            if tf.equal(y_true[i][j][2], 0):
                continue
            # 生成anchor_idxs的onehot向量 suchas:[False, False, True]
            # anchor_idxs:[6, 7, 8], [3, 4, 5], [0, 1, 2]中的一个
            # 例如：anchor_idxs=[6, 7, 8] y_true[i][j][5]=8
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))
            # tf.reduce_any()逻辑或 判断anchor_eq是否至少有一个True
            # True: IoU最大的anchor作为真实框
            if tf.reduce_any(anchor_eq):
                # 真实框的坐标：(ximn, ymin, xmax, ymax)
                box = y_true[i][j][0:4]
                # Bbox中心点坐标x_center=(xmin+xmax)/2, y_center=(ymin+ymax)/2
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2
                # 和真实框最拟合的anchoer索引 例如：anchor_idx=array([[2]]
                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                # 单位box_xy映射到grid：13*13;26*26;52*52
                # grid_xy：(x_center, y_center)*grid_size,真值框中点在特征图上的位置
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # 用于生成y_true_out的索引 
                # anchor_idx:特征图层的索引 共三层(13,26,52)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                # 用于生成y_true_out的值
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())
    # tf.tensor_scatter_nd_update(tensor, index, value):根据index和value更新tensor
    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())

# the output is tuple of shape
# ( [N, 13, 13, 3, 6],
#   [N, 26, 26, 3, 6],
#   [N, 52, 52, 3, 6] )
# N:batch_size   6:[x, y, w, h, obj, class]
def transform_targets(y_train, anchors, anchor_masks, size):
    # y_train：[batch_size, yolo_max_boxes, (xmin, ymin, xmax, ymax, label)]
    # N=batch_size, yolo_max_boxes=100
    y_outs = []
    # 网格大小： 32 16 8 下面更新公式:grid_size*2  
    grid_size = size // 32

    # calculate anchor index for true boxes
    # anchors数据类型转换
    anchors = tf.cast(anchors, tf.float32)
    # 每个anchor单位面积
    anchor_area = anchors[..., 0] * anchors[..., 1]
    # 真实框的[W, H]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    # tf.expand_dims(box_wh, -2): box_wh.shape:(N, 100, 2)=>(N, 100, 1, 2)
    # tf.tile():复制[w,h]tf.shape(anchors)[0]次. 
    # box_wh.shape:(N, 100, 1, 2)=>(N, 100, 9, 2)  9：9种anchor
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    # 真实框的面积 box_area.shape:(N, 100, 1, 9)
    box_area = box_wh[..., 0] * box_wh[..., 1]
    # 计算真实框box和anchors相交的面积并计算IoU
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    # 选取IoU值最大的anchor、计算真实框和哪个先验框最契合
    # anchor_idx.shape:(100, 1)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    # # anchor_idx.shape:(100, 1)=>(100, 1, 1)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    # y_train=>[N, 100, (xmin, ymin, xmax, ymax, label, anchor_idx)]
    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)

# 图像数据归一化到[0, 1] 加速收敛
def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
# 注释部分本项目内容不涉及,关于TFRecord读写可以参考tools.md文档。
IMAGE_FEATURE_MAP = {
    #'image/width': tf.io.FixedLenFeature([], tf.int64),
    #'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}

# 解析tfrecord格式数据的一个example：一张图片
def parse_tfrecord(tfrecord, class_table, size):
    # 以dict的方式返回IMAGE_FEATURE_MAP 定义的内容
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    # 将bytes数据重新编码成jpg图片
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    # tf.sparse.to_dense： 把SparseTensor转换为Tensor 效果类似SparseTensor.values
    class_text = tf.sparse.to_dense(x['image/object/class/text'], default_value='')

    # lookup在表中查找keys 输出相应的值，此处返回图片中name对应的行号 作为label
    # tf.cast(a, dtype)：张量数据类型转换 a的数据类型转换为dtype
    labels = tf.cast(class_table.lookup(class_text), tf.float32)

    # 堆叠：[[xmin[0], ymin[0], xmax[0], ymax][0],labels[0]], 
    #       [xmin[1], ymin[1], xmax[1], ymax[1], labels[1]], ...]  一张图片中可能有多个目标
    # y_train.shape:bbox_num*5
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    # 对y_train进行填充padding 即设定每张图片中有FLAGS.yolo_max_boxes个目标框，主要是为了后续的batch一致
    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train

# 加载TFRecord数据集
def load_tfrecord_dataset(file_pattern, class_file, size=416):
    LINE_NUMBER = -1  # 用行号作为键的值 
    # tf.lookup.TextFileInitializer： 从class_file进行初始化 LINE_NUMBER：从最后一行开始
    # 此处Key为20分类的name, value为行号
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    # 创建与模式匹配的所有文件的数据集 此处：TensorSpec [b'../data/voc2012_train.tfrecord']
    files = tf.data.Dataset.list_files(file_pattern)

    # 将数据集映射到TFRecordDataset函数并将结果展平
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))

# 没有填写数据集路径时，制作伪数据集
def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./data/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))


# 查看数据集单张图片内容
# file_pattern = './data/voc2012_train.tfrecord'
# class_file = './data/voc2012.names'
# size=416
def single_file_content(file_pattern, class_file, size=416):
    import matpoltlib.pyplot as plt
    from PIL import Image

    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, -1, delimiter="\n"), -1)
    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    tfrecord = dataset.take(1).as_numpy_iterator().next() 
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    text = x["image/object/class/text"].values.numpy()
    xmin = x["image/object/bbox/xmin"].values.numpy()
    image_encoded = x['image/encoded']
    img = tf.image.decode_jpeg(image_encoded, channels=3)
    plt.imshow(img)
    print(xmin, text)