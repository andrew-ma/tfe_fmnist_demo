#%%
### colab
# !pip install tensorflow-gpu==1.15.0
# !pip install tf-encrypted
#%%
import os

CPU_ONLY = False

if CPU_ONLY:
    # Set to CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import datetime
import tensorflow as tf

# disable deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if type(tf.contrib) != type(tf): tf.contrib._warning = None
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# assert tf.__version__
assert not str(tf.__version__).startswith('2'), "TF Encrypted only works with tensorflow 1"

if CPU_ONLY:
    assert not tf.test.is_gpu_available(), "CPU ONLY is not working"


if CPU_ONLY:
    training_device = "CPU"
else:
    if tf.test.is_gpu_available():
        training_device = "GPU"
    else:
        training_device = "CPU"

print("Using", training_device)

#%%
import tf_encrypted as tfe

#%%
from IPython import get_ipython
ipython = get_ipython()

def magic(magic_string):
    if '__IPYTHON__' in globals():
        ipython.magic(magic_string)
    else:
        print("Not running magic:", magic_string)

#%%
def cur_time_str():
    return datetime.datetime.now().strftime("%m%d%Y_%H%M%S")
#%%
magic('load_ext tensorboard')

TENSORBOARD_DIR = "tb"

tfe.set_tfe_trace_flag(True)
tfe.set_tfe_events_flag(True)
tfe.set_log_directory(TENSORBOARD_DIR)

CHECKPOINT_DIR = 'checkpoint'

#%%
tf.reset_default_graph()

#%%
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#%%
num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


#%%
batch_size = 128
epochs = 30

model = tf.keras.Sequential([
          tf.keras.layers.Conv2D(16, 8,
                                 strides=2,
                                 padding='same',
                                 activation='relu',
                                 input_shape=(28, 28, 1)),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Conv2D(32, 4,
                                 strides=2,
                                 padding='valid',
                                 activation='relu'),
          tf.keras.layers.AveragePooling2D(2, 1),
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(num_classes, activation='softmax')
  ])

print(model.summary())

#%%
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


# keras callbacks
cur_run_folder = cur_time_str() + "_" + training_device

# for generating summaries for use in tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join(TENSORBOARD_DIR, cur_run_folder))

# stops training early when it stops improving. Also useful for preventing overfitting
earlystopping_callback = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# for saving model regularly during training. Useful for deep learning that takes long time to train. Here we set to save only the best weights
cur_run_checkpoints_folder = os.path.join(CHECKPOINT_DIR, cur_run_folder)
if not os.path.exists(cur_run_checkpoints_folder): os.makedirs(cur_run_checkpoints_folder)
modelcheckpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(cur_run_checkpoints_folder, cur_run_folder + '_E{epoch:02d}_L{val_loss:0.3f}.h5'), save_best_only=True)

# # for changing learning rate while training, and it takes function with epoch_idx argument and returns new learning rate
# learningrate_callback = tf.keras.callbacks.LearningRateScheduler()

# log training metrics per epoch to CSV file
csvlogger_callback = tf.keras.callbacks.CSVLogger(os.path.join(cur_run_checkpoints_folder, cur_run_folder + '.csv'), append=True)

# train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[tensorboard_callback, modelcheckpoint_callback, earlystopping_callback, csvlogger_callback]
          )

# # load best weights if earlystopping disabled
# # find last saved checkpoint (will be best if save_best_only=True)
# best_weights_filename = sorted(os.listdir(os.path.join(cur_run_checkpoints_folder)))[0]
# model.load_weights(best_weights_filename)


# evaluate model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
