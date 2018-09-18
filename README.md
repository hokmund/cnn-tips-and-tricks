# cnn-tips-and-tricks
Code for AI Ukraine 2018 workshop "Tunning CNN: Tips &amp; Tricks"


To prepare for the workshop, please:

1. Clone code from this repository.

2. Download data, checkpoints and bottleneck features from http://tiny.cc/4flryy

3. Extract them from the archive and place under `src/`

&nbsp;&nbsp;&nbsp;&nbsp;At this point you should have such structure:
&nbsp;&nbsp;&nbsp;&nbsp;- checkpoints/
&nbsp;&nbsp;&nbsp;&nbsp;- images/
&nbsp;&nbsp;&nbsp;&nbsp;- test/
&nbsp;&nbsp;&nbsp;&nbsp;- train/
&nbsp;&nbsp;&nbsp;&nbsp;- validation/
&nbsp;&nbsp;&nbsp;&nbsp;- fine-tuning.ipynb
&nbsp;&nbsp;&nbsp;&nbsp;- data_analysis.ipynb
&nbsp;&nbsp;&nbsp;&nbsp;- pseudolabeling.ipynb
&nbsp;&nbsp;&nbsp;&nbsp;- clr.py
&nbsp;&nbsp;&nbsp;&nbsp;- demo_utils.py
&nbsp;&nbsp;&nbsp;&nbsp;- file_utils.py
&nbsp;&nbsp;&nbsp;&nbsp;- model_utils.py
&nbsp;&nbsp;&nbsp;&nbsp;- test_predictions.csv
&nbsp;&nbsp;&nbsp;&nbsp;- bottleneck_features_train.npy
&nbsp;&nbsp;&nbsp;&nbsp;- bottleneck_features_validation.npy
&nbsp;&nbsp;&nbsp;&nbsp;- requirements.txt

4. Run `pip install -r requirements.txt`
