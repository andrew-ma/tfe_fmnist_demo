import os
import argparse

CPU_ONLY = True

if CPU_ONLY:
    # Set to CPU only
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import datetime
from collections import OrderedDict
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
import tf_encrypted.keras.backend as KE

#%%
def filename_exists(x):
    if not os.path.exists(x):
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=filename_exists)
    parser.add_argument('--config_file', type=filename_exists, default='tfe.config')
    args = parser.parse_args()

    TENSORBOARD_DIR = "tb"

    tfe.set_tfe_trace_flag(True)
    tfe.set_tfe_events_flag(True)
    tfe.set_log_directory(TENSORBOARD_DIR)

    CHECKPOINT_DIR = 'checkpoint'

    # %%
    # load weights from previously trained model
    trained_model_filename = args.model_file
    assert os.path.exists(trained_model_filename), 'Trained model filename is invalid'

    # %%
    input_shape = (1, 28, 28, 1)
    output_shape = (1, 10)

    tf.reset_default_graph()

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 8,
                               strides=2,
                               padding='same',
                               activation='relu',
                               batch_input_shape=input_shape),  # input_shape => batch_input_shape
        tf.keras.layers.AveragePooling2D(2, 1),
        tf.keras.layers.Conv2D(32, 4,
                               strides=2,
                               padding='valid',
                               activation='relu'),
        tf.keras.layers.AveragePooling2D(2, 1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, name="logit")  # activation='softmax' => no final activation
        # returning logit instead of softmax since operation complex to perform using MPC (multi-party computation: the concept of shares where if you look at share on one server you get no info about original value - input data or model weights)
    ])

    print(model.summary())

    # %%
    model.load_weights(trained_model_filename)

    # %%
    # Configure protocol and the compute servers
    # using SecureNN protocol to secret share the model between each of 3 TFE servers, so we can provide predictions on encrypted data

    config_filename = args.config_file
    config = tfe.RemoteConfig.load(config_filename)

    tfe.set_config(config)
    tfe.set_protocol(tfe.protocol.SecureNN())

    # %%
    # convert tf keras model to tfe keras model
    tf.reset_default_graph()
    with tfe.protocol.SecureNN():
        tfe_model = tfe.keras.models.clone_model(model)

    print("Converted TF keras model to TFE keras model")
    # %%
    ##### setup new tfe.serving.QueueServer which will launch a serving queue so TFE compute servers can accept prediction requests on the secured model from external clients
    q_input_shape = input_shape
    q_output_shape = output_shape

    server = tfe.serving.QueueServer(
        input_shape=q_input_shape, output_shape=q_output_shape, computation_fn=tfe_model
    )
    print("Created Queue server to accept input {} and output {}".format(q_input_shape, q_output_shape))

    # %%
    sess = KE.get_session()

    # can set num_requests to set limit on number of predictions served by model
    request_limit = None

    request_ix = 1


    def step_fn():
        global request_ix
        print("Served encrypted prediction {i} to client".format(i=request_ix))
        request_ix += 1


    print("Running QueueServer with request limit {}".format(request_limit))

    server.run(
        sess,
        num_steps=request_limit,  # limit number of prediction requests
        step_fn=step_fn
    )
