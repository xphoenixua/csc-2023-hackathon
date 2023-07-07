import tensorflow as tf
from resnet50 import ResNet50
from attention import AttentionModel

layers = tf.keras.layers
reg = tf.keras.regularizers

_DECAY = 0.0001


class AutoencoderModel(tf.keras.Model):
    def __init__(self, reduced_dimension, expand_dimension, kernel_size=1,
               name='autoencoder'):
        super(AutoencoderModel, self).__init__(name=name)
        self.conv1 = layers.Conv2D(
            reduced_dimension,
            kernel_size,
            padding='same',
            name='autoenc_conv1')
        self.conv2 = layers.Conv2D(
            expand_dimension,
            kernel_size,
            activation=tf.keras.activations.relu,
            padding='same',
            name='autoenc_conv2')

        def call(self, inputs):
            dim_reduced_features = self.conv1(inputs)
            dim_expanded_features = self.conv2(dim_reduced_features)
            return dim_expanded_features, dim_reduced_features


class Delf(tf.keras.Model):
    def __init__(self,
                 block3_strides=True,
                 name='DELF',
                 pooling='avg',
                 gem_power=3.0,
                 embedding_layer=False,
                 embedding_layer_dim=2048,
                 use_dim_reduction=False,
                 reduced_dimension=128,
                 dim_expand_channels=1024):
        super(Delf, self).__init__(name=name)

        # Backbone using Keras ResNet50.
        self.backbone = ResNet50(
            'channels_last',
            name='backbone',
            include_top=False,
            pooling=pooling,
            block3_strides=block3_strides,
            average_pooling=False,
            gem_power=gem_power,
            embedding_layer=embedding_layer,
            embedding_layer_dim=embedding_layer_dim)

        # Attention model.
        self.attention = AttentionModel(name='attention')

        # Autoencoder model.
        self._use_dim_reduction = use_dim_reduction
        if self._use_dim_reduction:
            self.autoencoder = AutoencoderModel(reduced_dimension,
                                                dim_expand_channels,
                                                name='autoencoder')

    def init_classifiers(self, num_classes, desc_classification=None):
        self.num_classes = num_classes
        if desc_classification is None:
            self.desc_classification = layers.Dense(
                num_classes, activation=None, kernel_regularizer=None, name='desc_fc')
        else:
            self.desc_classification = desc_classification
        self.attn_classification = layers.Dense(
            num_classes, activation=None, kernel_regularizer=None, name='att_fc')

    def forward_pass(self, images, training=True):
        backbone_blocks = {}
        block3 = backbone_blocks['block3']
        block3 = tf.stop_gradient(block3)
        if self._use_dim_reduction:
            (dim_expanded_features, dim_reduced_features) = self.autoencoder(block3)
            attn_prelogits, attn_scores, _ = self.attention(
                block3,
                targets=dim_expanded_features,
                training=training)
        else:
            attn_prelogits, attn_scores, _ = self.attention(block3, training=training)
            dim_reduced_features = None
        return (attn_scores, backbone_blocks, dim_reduced_features)

    def build_call(self, input_image, training=True):
        (attn_scores, backbone_blocks, dim_reduced_features) = self.forward_pass(input_image, training)
        if self._use_dim_reduction:
            features = dim_reduced_features
        else:
            features = backbone_blocks['block3']
        return attn_scores, features

    def call(self, input_image, training=True):
        probs, features = self.build_call(input_image, training=training)
        return probs, features