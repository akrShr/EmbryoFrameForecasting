from DataPreparation import get_data
from Model import get_model
from tensorflow import keras
import matplotlib.pyplot as plt

model = get_model('cells')
train_x, train_y, val_x, val_y = get_data()

model.compile(
  loss=keras.losses.binary_crossentropy,
  optimizer=keras.optimizers.Adam(clipvalue=1.0),
)

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience = 10)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience = 5)

epochs = 10
batch_size = 2

history = model.fit(train_x,train_y,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (val_x, val_y),
            callbacks = [early_stopping, reduce_lr])

epochs = range(1, len(history.history["loss"]) + 1)
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure()
plt.plot(epochs, loss, "r", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

