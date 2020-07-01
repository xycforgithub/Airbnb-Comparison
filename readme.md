# Crowdsourcing Comparisons and Labels for Airbnb Listing Price Estimation

This repository contains the crowdsourced data for Airbnb listing price comparison as described in:

Yichong Xu, Sivaraman Balakrishnan, Aarti Singh and Artur Dubrawski.<br/>
Regression with Comparisons: Escaping the Curse of Dimensionality with Ordinal Information</br>
arXiv preprint arXiv:1806.03286, 2018<br/>
[arXiv version](https://arxiv.org/abs/1809.06963)<br/>
[Conference version in ICML 2018](http://proceedings.mlr.press/v80/xu18e.html) 

Please cite the above paper if you use this data. 

We collected comparisons and direct labels for evaluating the Airbnb listing prices in Seattle, Washington, US. We record the crowdsource workers' answers, as well as their response time. The Airbnb listing data is from [Kaggle](https://www.kaggle.com/AirBnB/seattle/home).

## Data description
`raw_data.json` contains the raw data including:<br/>
	> `features`: Textual and numerical features for each listing.<br/>
	> `labels`: For each listing, we collect 5 (for training set) or 10 (for test set) labels from crowdsource workers.<br/>
	> `comparisons`: We additionally collect comparisons between 1,895 pairs of listings. We collect two comparisons for each pair. The `pair` entry contains the indices of the pairs, and `data` entry contains the comparisons that we have collected. All comparisons are on the training set.<br/>
	> `num_train_data`: Number of training data (we have 389 training points and 97 testing points).<br/>

`generate_data.py` featurize the raw data into a numerical matrix that can be used by subsequent algorithms. We use 16 features in total (55 if expanding categorical features to binary ones), as described in [our paper](https://arxiv.org/abs/1809.06963) and the Python script. For convenience, we include the processing result in `vectorized_data.json`.


by Yichong Xu

xuyc11@gmail.com



