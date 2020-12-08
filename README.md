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