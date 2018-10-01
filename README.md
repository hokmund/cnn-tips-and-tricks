# cnn-tips-and-tricks
Code for AI Ukraine 2018 workshop "Tunning CNN: Tips &amp; Tricks"


To prepare for the workshop, please:

1. Clone code from this repository.

2. Download data, checkpoints and bottleneck features from http://tiny.cc/4flryy 

   There are 3 Gb of data and checkpoints, so be sure to download and unzip them before the conference.

3. Extract them from the archive and place under `src/`

   At this point you should have such structure:
   - checkpoints/
   - images/
   - test/
   - train/
   - validation/
   - fine-tuning.ipynb
   - data_analysis.ipynb
   - pseudolabeling.ipynb
   - clr.py
   - demo_utils.py
   - file_utils.py
   - model_utils.py
   - test_predictions.csv
   - bottleneck_features_train.npy
   - bottleneck_features_validation.npy
   - requirements.txt

4. Run `pip install -r requirements.txt`


Code was developed and tested on Ubuntu 16.04 and Python 3.6. However, it should work on Windows and Mac OS as well.

If you want to speed-up computations in this workshop, but you don't have enough CPU/GPU, please try Google Colab Cloud TPU (https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/shakespeare_with_tpu_and_keras.ipynb).