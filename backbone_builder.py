from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
import keras

def build_backbone(R_input):    
    backbone = EfficientNetV2B0(
        include_top=False, input_shape=[R_input, R_input, 3]
    )
    list_layer=[layer.name for layer in backbone.layers]
    c3_idx = list_layer.index('block4a_expand_conv')-1
    c4_idx = list_layer.index('block5a_expand_conv')-1
    #c5_idx = list_layer.index('top_activation')-3

    c3_output, c4_output, c5_output = [
        backbone.get_layer(index=c3_idx).output,
        backbone.get_layer(index=c4_idx).output,
        backbone.get_layer(name='top_activation').output
        #backbone.get_layer(index=c5_idx).output,
    ]
    
    backbone = keras.Model(inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output])

    for layer in backbone.layers:
      if 'stem' in layer.name or isinstance(layer, keras.layers.BatchNormalization):
        layer.trainable = False
    #backbone.trainable = False
      
    return backbone