import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from UnetModel import get_model


def get_emb_images():
    img_size = (128,128)
    emb_files = glob('EmbCropper_Data/embryoImages/*.jpg')
    num_imgs = len(emb_files)
    input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype='float32')
    i=0
    for emb_file in emb_files:
        input_imgs[i] = img_to_array(load_img(emb_file,target_size=img_size))/255
        i += 1
    return input_imgs

def get_emb_masks():
    img_size = (128, 128)
    emb_files = glob('EmbCropper_Data/maskImages/*.png')
    num_imgs = len(emb_files)
    input_imgs = np.zeros((num_imgs,) + img_size + (1,), dtype='uint8')
    i = 0
    for emb_file in emb_files:
        input_imgs[i] = img_to_array(load_img(emb_file, target_size=img_size, color_mode="grayscale")) / 255
        i += 1
    return input_imgs

emb_imgs = get_emb_images()
targets = get_emb_masks()
train_input_imgs, val_input_imgs, train_targets, val_targets = train_test_split(emb_imgs, targets, test_size=0.2, random_state=42)

unet_model = get_model()
unet_model.compile(optimizer="adam",
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('Weights/EmbCropper.keras',
                                       monitor='val_loss',
                                       mode='min',
                                       save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                     patience=5,
                                     mode="min")
]

history = unet_model.fit(train_input_imgs, train_targets,
                         epochs=50,
                         callbacks=callbacks,
                         batch_size=32,
                         validation_data=(val_input_imgs, val_targets))

epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure()
plt.plot(epochs, loss, "r", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()