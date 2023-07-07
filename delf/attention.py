import tensorflow as tf

layers = tf.keras.layers
reg = tf.keras.regularizers

_DECAY = 0.0001


class AttentionModel(tf.keras.Model):
    def __init__(self, kernel_size=1, decay=_DECAY, name='attention'):
        super(AttentionModel, self).__init__(name=name)

        # Перший згортковий шар (ReLU)
        self.conv1 = layers.Conv2D(
            512,
            kernel_size,
            kernel_regularizer=reg.l2(decay),
            padding='same',
            name='attn_conv1')
        self.bn_conv1 = layers.BatchNormalization(axis=3, name='bn_conv1')

        # Другий згортковий шар (SoftPlus)
        self.conv2 = layers.Conv2D(
            1,
            kernel_size,
            kernel_regularizer=reg.l2(decay),
            padding='same',
            name='attn_conv2')
        self.activation_layer = layers.Activation('softplus')

    def call(self, inputs, targets=None, training=True):
        x = self.conv1(inputs)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)

        score = self.conv2(x)
        prob = self.activation_layer(score)

        # агрегуємо вхідні дані, якщо targets - None
        if targets is None:
            targets = inputs

        # l2-нормалізація для featuremap перед пулінгом
        targets = tf.nn.l2_normalize(targets, axis=-1)
        feat = tf.reduce_mean(tf.multiply(targets, prob), [1, 2], keepdims=False)

        return feat, prob, score