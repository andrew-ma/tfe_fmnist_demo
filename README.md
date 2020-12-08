https://github.com/tf-encrypted/tf-encrypted


Install
```
conda create -n tfe python=3.6
conda activate tfe
conda install tensorflow-gpu=1.15.0 notebook jupyter
python3 -m pip install tf-encrypted
# python3 -m pip install tensorflow-privacy
```

If running locally, run this 

```tensorboard --logdir tb --reload_multifile True```

To host metrics publicly, run

```tensorboard dev upload --logdir tb```

## private prediction
* main_train_model_on_plaintext.py is for training normal fmnist model
* the model will be converted to a TFE model in main_tfe_prediction_server.py
* Compute servers will be configured and run in separate terminals
* a QueueServer will be setup using the TFE model with known input shape and output shape
* Prediction clients will create QueueClient to connect to QueueServer and upload their prediction input split up into shares to the multiple compute servers, which will run input data share through model weights and return back result share. Result shares will be combined to the prediction.


Command for running each compute server

```python -m tf_encrypted.player --config tfe.config {key_in_config_file}```


### tests were run with checkpoint/12082020_063944_GPU/12082020_063944_GPU_E27_L0.293.h5 model weights


https://tensorboard.dev/experiment/Jv0NnD9tRECQA6GU8RBYTA/
