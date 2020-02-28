from absl import app, flags, logging
from absl.flags import FLAGS

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('weights_num_classes', None, 'specify num class for `weights` file if different, '
                     'useful in transfer learning with different number of classes')

# 'none: 从0开始训练, 参数随机初始化, '
# 'darknet: 冻结darknet参数, 其余参数随机初始化'
# 'no_output: yolo_ouput层参数随机初始化,冻结其余参数, '
# 'frozen: 冻结所有参数,训练不起作用, '
# 'fine_tune: 冻结darknet参数, 其余参数在预训练权重的基础上训练')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        # 设置仅在需要时申请显存空间
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 判断训练tiny版本的YOLO还是完整版的YOLO
    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # 如果未指定数据集则加载一张图片作为数据集=>fake_dataset
    train_dataset = dataset.load_fake_dataset()

    # 判断数据集路径是否为空
    if FLAGS.dataset:
        # 从TFRecode文件加载数据集 train_dataset：(x_train, y_train)
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes, FLAGS.size)
    # 生成批训练数据
    # 打乱数据顺序
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    # y.shape:train_dataset.as_numpy_iterator().next()[1].shape 
    # =>(batch_size, yolo_max_boxes, 5) 5=>(xmin, ymin, xmax, ymax, classlabel)
    train_dataset = train_dataset.map(lambda x, y: (
        # 图像数据归一化[0,1]
        dataset.transform_images(x, FLAGS.size),
        # 根据先验框anchor确定bbox属于哪一层特征图(13*13, 26*26, 52*52) 
        # 并计算出bbox的中心点在特征图上的位置
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))
    # 数据预读取，提高延迟和吞吐量
    # tf.data.experimental.AUTOTUNE：根据可用CPU动态设置并行调用的数量
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    # 加载伪验证集，防止没有添加验证集路径时报错
    val_dataset = dataset.load_fake_dataset()
    # 加载验证集  解释同上
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, FLAGS.size)))

    # Configure the model for transfer learning
    # 训练模式选择
    # 随机初始化权重，从0开始训练整个网络
    if FLAGS.transfer == 'none':
        pass  # Nothing to do
    # 迁移训练的两种方式
    elif FLAGS.transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if FLAGS.tiny:
            model_pretrained = YoloV3Tiny(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        else:
            # 模型网络结构
            model_pretrained = YoloV3(
                FLAGS.size, training=True, classes=FLAGS.weights_num_classes or FLAGS.num_classes)
        # 加载预训练权重
        model_pretrained.load_weights(FLAGS.weights)

        # 设置darknet网络权重并冻结网络,即主干网络不参与训练,其余参数随机初始化
        if FLAGS.transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        # 设置YOLO输出层以外的网络的权重并冻结, 即仅训练YOLO的输出层且参数随机初始化
        elif FLAGS.transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)

    # 迁移学习fine_tune和frozen模式要求训练的类别数和预训练权重一致(80类)
    else:
        # All other transfer require matching classes
        # 加载网络所有预训练权重参数
        model.load_weights(FLAGS.weights)
        # 冻结darknet(骨干网络)权重, 其余参数在预训练权重的基础上训练
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        # 冻结所有参数,训练不起作用.
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)
            
    # 定义优化器:Adam
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)

    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]

    # 调试模型:速度慢:  Eager: op 在调用后会立即运行
    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        # 训练集上的平均loss/验证集上的平均loss
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        # 迭代每个epoch
        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                # 梯度带:自动计算变量梯度
                with tf.GradientTape() as tape:
                    # model(): eager模式下选择此方式,不需要编译直接运行, 速度快.
                    # model.predict()第一次运行时需要先编译图模式
                    outputs = model(images, training=True)
                    # 计算张量各维度的元素之和.
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss
                # 梯度
                grads = tape.gradient(total_loss, model.trainable_variables)
                # 执行最优化器
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                # 记录日志文件
                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                # 更新平均loss
                avg_loss.update_state(total_loss)

            # 在验证集上验证
            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)
            # .result()：返回累计结果
            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))
            # reset_states：清除累计值
            avg_loss.reset_states()
            avg_val_loss.reset_states()
            # 每个epoch保存一次模型权重
            model.save_weights(
                'checkpoints/yolov3_train_{}.tf'.format(epoch))
    
    # 训练模式
    else:
        # 编译模型
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'],
                      run_eagerly=(FLAGS.mode == 'eager_fit'))
        # 回调函数
        callbacks = [
            # lr衰减
            ReduceLROnPlateau(verbose=1),
            # lr不变时停止训练
            EarlyStopping(patience=3, verbose=1),
            # 保存模型
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            # 训练结果可视化
            TensorBoard(log_dir='logs', write_images=True, update_freq='batch')
        ]
        # 进行迭代训练
        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
