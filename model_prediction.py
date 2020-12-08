#%%
import os
import argparse

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
def cur_time_str():
    return datetime.datetime.now().strftime("%m%d%Y_%H%M%S")

cur_run_folder = cur_time_str() + "_" + training_device

#%%
def filename_exists(x):
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=filename_exists)
    args = parser.parse_args()

    # load weights from previously trained model
    trained_model_filename = args.model_file
    assert os.path.exists(trained_model_filename), 'Trained model filename is invalid'

    model = tf.keras.models.load_model(trained_model_filename)

    # %%
    from tensorflow.keras.datasets import fashion_mnist

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

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



    print("Evaluating on test dataset")
    import time

    # batch size 1
    print("Batch size 1")
    start_time = time.time()
    score = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
    test_time_1 = time.time() - start_time
    print(dict(zip(model.metrics_names, score)))
    print("Time", test_time_1)
    results_1 = {
    'Loss (batchsize=1)': str(score[0]),
    'Accuracy (batchsize=1)': str(score[1]),
        'Time (batchsize=1)': str(test_time_1)
    }


    TEST_RESULTS_DIR = 'tests'
    if not os.path.exists(TEST_RESULTS_DIR): os.makedirs(TEST_RESULTS_DIR)
    test_output_filename = os.path.join(TEST_RESULTS_DIR, cur_run_folder + '_test_results.json')

    import json
    json.dump(results_1, open(test_output_filename, 'w'))