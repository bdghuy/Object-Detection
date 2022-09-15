import tensorflow as tf
import keras
from tensorflow.keras.layers import UpSampling2D, Reshape, AveragePooling2D
from tensorflow.keras.activations import swish
from tensorflow_addons.layers import GroupNormalization
import numpy as np

from backbone_builder import build_backbone 
from anchors import AnchorBox
from utils import convert_to_corners

class WSConv2D(keras.layers.Conv2D):
    def call(self, inputs):
        mean, var = tf.nn.moments(self.kernel, axes=[0, 1, 2], keepdims=True)
        h, w, c_in, c_out = self.kernel.shape
        fan_in = h*w*c_in
        result = self.convolution_op(
            inputs, (self.kernel - mean) / tf.sqrt(var*fan_in + 1e-10)
        )
        if self.use_bias:
            result = result + self.bias
        return result


def conv_gnorm_act(input_tensor, filters, kernel_size=3, strides=1, padding='same', use_sepconv=True, use_act=True):
  if use_sepconv:
    #x = WSSeparableConv2D(filters, kernel_size, strides, padding, depthwise_initializer=tf.keras.initializers.VarianceScaling(), pointwise_initializer=tf.keras.initializers.VarianceScaling())(input_tensor)
    x = tf.keras.layers.SeparableConv2D(filters, kernel_size, strides, padding, depthwise_initializer=tf.keras.initializers.VarianceScaling(), pointwise_initializer=tf.keras.initializers.VarianceScaling())(input_tensor)
  else:
    x = WSConv2D(filters, kernel_size, strides, padding, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(input_tensor)
    #x = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))(input_tensor)
  x = GroupNormalization(groups=4)(x)
  if use_act:
    return swish(x)
  return x


def resample_feature_map(feat, target_height, target_width, target_num_channels):
  _, height, width, num_channels = feat.shape
  # channel
  if num_channels != target_num_channels:
    feat = conv_gnorm_act(feat, filters=target_num_channels, kernel_size=1, use_sepconv=False, use_act=False)
    
  # spatial
  if height > target_height and width > target_width:
    # Downsample
    height_stride_size = int((height - 1) // target_height + 1)
    width_stride_size = int((width - 1) // target_width + 1)
    feat = AveragePooling2D(pool_size=(height_stride_size, width_stride_size), strides=(height_stride_size, width_stride_size), padding="same")(feat)
  
  elif height <= target_height and width <= target_width:
    if height < target_height or width < target_width:
      # Upsample
      feat = tf.compat.v1.image.resize(feat, (target_height, target_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  return feat


class fuse_features(keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(fuse_features, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(fuse_features, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config


def build_bifpn_layer(feats, channels):
  # P7 (4)---------------P7_out (12)
  # |         \            |
  # P6 (3)---P6_td (5)---P6_out (11)
  # |          |           |
  # P5 (2)---P5_td (6)---P5_out (10)
  # |          |           |
  # P4 (1)---P4_td (7)---P4_out (9)
  # |            \         |
  # P3 (0)---------------P3_out (8)
  
  bifpn_dict = [
                {'node':5, 'inputs_offsets':[3, 4]},
                {'node':6, 'inputs_offsets':[2, 5]},
                {'node':7, 'inputs_offsets':[1, 6]},
                {'node':8, 'inputs_offsets':[0, 7]},
                {'node':9, 'inputs_offsets':[1, 7, 8]}, #[1, 7, 8]
                {'node':10, 'inputs_offsets':[2, 6, 9]}, #[2, 6, 9]
                {'node':11, 'inputs_offsets':[3, 5, 10]}, #[3, 5, 10]
                {'node':12, 'inputs_offsets':[4, 11]}
               ]
  
  for node in bifpn_dict:
    _, target_height, target_width, _ = feats[node['inputs_offsets'][0]].shape

    input_nodes = []
    for input_offset in node['inputs_offsets']:
      input_node = resample_feature_map(feats[input_offset], target_height, target_width, channels)
      input_nodes.append(input_node)
      
    new_node = fuse_features()(input_nodes)
    new_node = conv_gnorm_act(new_node, filters=channels, use_sepconv=False)
    feats.append(new_node)
  
  new_feats = feats[-5:]

  return new_feats


def build_feature_network(feats, channels, repeats):
  # Build additional input features that are not from backbone.
  for _ in range(2):
    # Adds a coarser level by downsampling the last feature map.
    feats.append(resample_feature_map(feats[-1], 
                                      target_height=(feats[-1].shape[1] - 1) // 2 + 1, 
                                      target_width=(feats[-1].shape[2] - 1) // 2 + 1, 
                                      target_num_channels=channels))
  
  for rep in range(repeats):
    feats = build_bifpn_layer(feats, channels)
  
  return feats


class ClassNet(keras.Model):
  def __init__(self, num_classes, num_filters, repeats, num_anchors=9, **kwargs):
    super(ClassNet, self).__init__(**kwargs)
    self.level = 3

    self.num_classes = num_classes
    self.repeats = repeats
    self.convs = [WSConv2D(filters=num_filters, 
                                        kernel_size=3, 
                                        padding='same', 
                                        name='class-%d'%i) for i in range(repeats)]
    self.gns = {level:[GroupNormalization(groups=4, name='class-%d-gn-%d'%(i, level)) for i in range(repeats)] 
                for level in range(3,8)}
    self.classes = WSConv2D(filters=num_classes*num_anchors, 
                                         kernel_size=3, 
                                         padding='same', 
                                         name='class-predict', 
                                         bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)))
    self.act = swish
    self.reshape = Reshape((-1, num_classes))
  
  def call(self, inputs, **kwargs):
    feature = inputs
    for i in range(self.repeats):
      feature = self.convs[i](feature)
      feature = self.gns[self.level][i](feature)
      feature = self.act(feature)
    
    if self.level < 7:
      self.level += 1
    else:
      self.level = 3

    classes = self.classes(feature)
    classes = self.reshape(classes)
    
    return classes


class BoxNet(keras.Model):
  def __init__(self, num_filters, repeats, num_anchors=9, **kwargs):
    super(BoxNet, self).__init__(**kwargs)
    self.level = 3

    self.repeats = repeats
    self.convs = [WSConv2D(filters=num_filters, 
                                        kernel_size=3, 
                                        padding='same', 
                                        name='box-%d'%i) for i in range(repeats)]
    self.gns = {level:[GroupNormalization(groups=4, name='box-%d-gn-%d'%(i, level)) for i in range(repeats)]
                for level in range(3,8)}
    self.boxes = WSConv2D(filters=4*num_anchors, 
                                       kernel_size=3, 
                                       padding='same', 
                                       name='box-predict')
    self.act = swish
    self.reshape = Reshape((-1, 4))
  
  def call(self, inputs, **kwargs):
    feature = inputs
    for i in range(self.repeats):
      feature = self.convs[i](feature)
      feature = self.gns[self.level][i](feature)
      feature = self.act(feature)

    if self.level < 7:
      self.level += 1
    else:
      self.level = 3
    
    boxes = self.boxes(feature)
    boxes = self.reshape(boxes)
    
    return boxes


class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=80,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, predictions):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )


def detector(R_input, num_classes, W_bifpn, D_bifpn, D_head):
  backbone = build_backbone(R_input)
  fpn_feats = build_feature_network(feats=backbone.output, channels=W_bifpn, repeats=D_bifpn)
  cls_net = ClassNet(num_classes=num_classes, num_filters=W_bifpn, repeats=D_head)
  box_net = BoxNet(num_filters=W_bifpn, repeats=D_head)

  cls_outputs = []
  box_outputs = []
  for feature in fpn_feats:
    cls_outputs.append(cls_net(feature))
    box_outputs.append(box_net(feature))

  cls_outputs = tf.concat(cls_outputs, axis=1)
  box_outputs = tf.concat(box_outputs, axis=1)
  outputs = tf.concat([box_outputs, cls_outputs], axis=-1)

  model = keras.Model(inputs=[backbone.input], outputs=[outputs])
  return model


def infer_detector(R_input, num_classes, W_bifpn, D_bifpn, D_head, conf_threshold, iou_threshold, weights_path):
  model = detector(R_input, num_classes, W_bifpn, D_bifpn, D_head)
  model.load_weights(weights_path)
  image = tf.keras.Input(shape=[R_input, R_input, 3], name="image")
  predictions = model(image, training=False)
  detections = DecodePredictions(num_classes=num_classes, confidence_threshold=conf_threshold, nms_iou_threshold=iou_threshold)(image, predictions)
  infer_model = tf.keras.Model(inputs=image, outputs=detections)

  return infer_model