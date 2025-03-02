import tensorflow as tf

def double_conv_block(x, n_filters):
  x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
  x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
  return x

def downsample_block(x, n_filters):
  f = double_conv_block(x, n_filters)
  p = tf.keras.layers.MaxPool2D(2)(f)
  p = tf.keras.layers.Dropout(0.5)(p)
  return f, p

def upsample_block(x, conv_features, n_filters):
  x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
  if conv_features.shape[1] != x.shape[1]:
    limit = x.shape[1]
    conv_features = conv_features[:, 0:limit, 0:limit, :]
  x = tf.keras.layers.concatenate([x, conv_features])
  x = tf.keras.layers.Dropout(0.5)(x)
  x = double_conv_block(x, n_filters)
  return x


def get_model():
  inputs = tf.keras.Input(shape=(128, 128, 3))
  # flipping horizontally and vertically to augment data
  data_augment = tf.keras.layers.RandomFlip(seed=42)(inputs)

  # encoder
  f1, p1 = downsample_block(data_augment, 32)
  f2, p2 = downsample_block(p1, 64)
  f3, p3 = downsample_block(p2, 128)
  f4, p4 = downsample_block(p3, 256)

  bottleneck = double_conv_block(p4, 512)

  # decoder
  u6 = upsample_block(bottleneck, f4, 256)
  u7 = upsample_block(u6, f3, 128)
  u8 = upsample_block(u7, f2, 64)
  u9 = upsample_block(u8, f1, 32)

  outputs = tf.keras.layers.Conv2D(2, 1, padding="same", activation = "sigmoid")(u9)

  unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
  print(unet_model.summary())

  return unet_model


