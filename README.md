# cnn-tips-and-tricks
Code for AI Ukraine 2018 workshop "Tunning CNN: Tips &amp; Tricks"


To prepare for the workshop, please:

1. Clone code from this repository.

2. Download data, checkpoints and bottleneck features from http://tiny.cc/4flryy

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


Code was developed and tested on Ubuntu 16.04 and Python 3.6. However, it should work on Windows and Mac OS with Python 3.4+ versions as well.
