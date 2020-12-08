import os
import argparse
import numpy as np

CPU_ONLY = True

if CPU_ONLY:
    # Set to CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import datetime
from collections import OrderedDict
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


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
    parser.add_argument('--config_file', type=filename_exists, default='tfe.config')
    args = parser.parse_args()

    TENSORBOARD_DIR = "tb"

    tfe.set_tfe_trace_flag(True)
    tfe.set_tfe_events_flag(True)
    tfe.set_log_directory(TENSORBOARD_DIR)

    CHECKPOINT_DIR = 'checkpoint'

    # %%

    input_shape = (1, 28, 28, 1)
    output_shape = (1,10)

    # Configure protocol and the compute servers
    # using SecureNN protocol to secret share the model between each of 3 TFE servers, so we can provide predictions on encrypted data

    config_filename = args.config_file
    config = tfe.RemoteConfig.load(config_filename)

    tfe.set_config(config)
    tfe.set_protocol(tfe.protocol.SecureNN())

    client = tfe.serving.QueueClient(
        input_shape=input_shape,
        output_shape=output_shape
    )

    sess = tfe.Session(config=config)

    #%%
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    fashion_mnist_labels = ["T-shirt/top",  # index 0
                            "Trouser",  # index 1
                            "Pullover",  # index 2
                            "Dress",  # index 3
                            "Coat",  # index 4
                            "Sandal",  # index 5
                            "Shirt",  # index 6
                            "Sneaker",  # index 7
                            "Bag",  # index 8
                            "Ankle boot"]  # index 9
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # %%
    TEST_RESULTS_DIR = 'tests'
    test_results_filename = os.path.join(TEST_RESULTS_DIR, cur_run_folder + 'TFE_test_results.json')
    # record time for predicting all the test data
    # record the accuracy
    num_correct = 0

    import time

    start_time = time.time()
    for image, expected_label in zip(x_test, y_test):

        # client.run will insert the image into the queue, secret share the data locally, and submit the shares to the compute server
        res = client.run(
            sess,
            image.reshape(*input_shape)
        )

        predicted_label = np.argmax(res)

        if expected_label == predicted_label:
            num_correct += 1

        # print("Image had label {} and was {} classified as {}".format(expected_label,
        #     "correctly" if expected_label == predicted_label else "incorrectly",
        #     predicted_label))

    test_time_1 = time.time() - start_time

    accuracy = num_correct / len(y_test)
    print("Accuracy (batchsize=1)", accuracy, "{}/{} correct".format(num_correct, len(y_test)))
    print("Time", test_time_1)

    results = {'Accuracy': str(accuracy), 'Time (batchsize=1)': str(test_time_1)}

    import json
    json.dump(results, open(test_results_filename, 'w'))