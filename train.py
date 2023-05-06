
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.metrics import Recall, Precision
tf.debugging.set_log_device_placement(True)

from data import load_data, tf_dataset
from model import build_model

# Check if TensorFlow is using GPU
if tf.test.is_gpu_available():
    print("Training on GPU")
else:
    print("Training on CPU")
    
def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

if __name__ == "__main__":
    ## seeding
    np.random.seed(42)
    tf.random.set_seed(42)
    
    ## Dataset
    path = "C:/Users/mahsh/OneDrive/Bureau/inner_circle/train-results"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)
    
    ## Hyperparameters
    batch = 8
    lr = 1e-4
    epochs = 100

    train_dataset = tf_dataset(train_x, train_y, batch=batch)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch)

    model = build_model()

    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    callbacks = [
        ModelCheckpoint("files/model.h5"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    train_steps = len(train_x)//batch
    valid_steps = len(valid_x)//batch

    if len(train_x) % batch != 0:
        train_steps += 1
    if len(valid_x) % batch != 0:
        valid_steps += 1

    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks,
        shuffle= False
        )
    
    
########################################

    
# import pandas as pd
# import matplotlib.pyplot as plt
# import random
# import warnings
# from keras.preprocessing.image import load_img

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# epochs = range(len(acc))

# plt.plot(epochs, acc, 'b', label= 'Training Accuracy')
# plt.plot(epochs, val_acc, 'r' , label= 'validation Accuracy')
# plt.title('Accuracy Graph')
# plt.legend()
# plt.figure()

# loss = history.history['loss']
# val_loss = history.history['val_loss']
# plt.plot(epochs, loss, 'b', label= 'Training Loss')
# plt.plot(epochs, val_loss, 'r' , label= 'validation Loss')
# plt.title('Loss Graph')
# plt.legend()
# plt.show()

# recall = history.history['recall']
# val_recall = history.history['val_recall']
# plt.plot(epochs, recall, 'b', label= 'Training recall')
# plt.plot(epochs, val_recall, 'r' , label= 'validation recall')
# plt.title('recall Graph')
# plt.legend()
# plt.show()

# precision = history.history['precision']
# val_precision = history.history['val_precision']
# plt.plot(epochs, precision, 'b', label= 'Training precision')
# plt.plot(epochs, val_precision, 'r' , label= 'validation precision')
# plt.title('precision Graph')
# plt.legend()
# plt.show()


# import pandas as pd
# import seaborn as sns

# # convert the data to a DataFrame
# history_df = pd.DataFrame(history.history)
# history_df['epoch'] = range(len(history_df))

# # plot the accuracy
# sns.set_style('darkgrid')
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=history_df, x='epoch', y='acc', label='Training Accuracy')
# sns.lineplot(data=history_df, x='epoch', y='val_acc', label='Validation Accuracy')
# plt.title('Accuracy Graph')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
