# fake-review-classifier
Fake reviews are a growing problem in today's online world. This project is an attempt to answer the question of how to detect fake reviews (spam). It contains the code to train a classifier on Amazon product data, and use this model to classify whether Yelp reviews are real or fake. It was built by Kevin Zhang, Rushi Gajaria, and Aaqib Mujtaba as the term project for the Fall 2020 offering of CS486, Introduction to Articial Intelligence, at the University of Waterloo.

### Model Specifications
As part of the experiment design, we proposed one full model, which represents our hypothesis, and two other models for comparison. The full model represents a repeated k-means classifier. We choose to use 100 repetitions since this has been shown to effectively reduce the impact of errors due to cluster initialization. We also choose to use [k-means++](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) for cluster initialization to further minimize the chances of bad clustering.

We convert a review into four features: length of review text, days since product posting, agreement between sentiment and product rating, and a novel feature the we called sentiment variability. Length of review text refers to the number of characters in the body text of a review - studies have shown that real reviews have longer body texts. Studies have also shown that more fake reviews are posted closer to the date at which a product was posted, since this is when the fake reviews can have a larger influence on customer decision making. For agreement between sentiment and product rating, we first compute the sentiment of the body text of the review, then find the difference between the scaled sentiment (1 to 5) and the rating. Previous work has shown that real reviews have stronger agreement. Finally, sentiment variability is defined as the variation between the sentiments of sentences in a review. It is expected that real reviews will have higher variability, as real users should be less biased.

### Dependencies
The following dependencies are required to run the program:

- VADER: `pip3 install vaderSentiment --user`

- Textblob: `pip3 install -U textblob --user`

- Scikit-Learn: `pip3 install sklearn --user`

- Pandas: `pip3 install pandas`

- Nltk: `pip3 install --user -U nltk`

- Numpy: `pip install --user -U numpy`

### Running the program
Four programs are provided to run the model: `process_datasets.py`, `test_full_model.py`, `test_half_model.py`, and `test_naive_model.py`. 

The first, `process_datasets.py`, takes in all the data that will be used, and converts it to the feature vectors that are actually used for training, validation, and testing. Since these datasets are large, this can take multiple hours. After running `python3 process_datasets.py`, the output should look like:
```
[nltk_data] Downloading package punkt to /Users/kzzhang/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package stopwords to
[nltk_data]     /Users/kzzhang/nltk_data...
[nltk_data]   Package stopwords is already up-to-date!
[nltk_data] Downloading package wordnet to /Users/kzzhang/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package sentiwordnet to
[nltk_data]     /Users/kzzhang/nltk_data...
[nltk_data]   Package sentiwordnet is already up-to-date!
[nltk_data] Downloading package wordnet to /Users/kzzhang/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /Users/kzzhang/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
Parsing datasets
Training dataset size: 20260 entries
Validation dataset size: 67395 entries
Testing dataset size: 89763 entries
Processing Training Dataset
===================
	preprocessing training dataset
	building training feature map
		handling VADER
		handling TextBlob
		handling SentiWordNet
	saving training features
Processing Validation Dataset
===================
	preprocessing validation dataset
	getting labels
	building validation feature map
		handling VADER
		handling TextBlob
		handling SentiWordNet
	saving validation features
Processing Test Dataset
===================
	preprocessing testing dataset
	getting labels
	building testing features
		handling VADER
		handling TextBlob
		handling SentiWordNet
	saving testing features
Total execution time: 4:41:29.858038
```
