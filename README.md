# fake-review-classifier
Fake reviews are a growing problem in today's online world. This project is an attempt to answer the question of how to detect fake reviews (spam). It contains the code to train a classifier on Amazon product data, and use this model to classify whether Yelp reviews are real or fake. It was built by Kevin Zhang, Rushi Gajaria, and Aaqib Mujtaba as the term project for the Fall 2020 offering of CS486, Introduction to Articial Intelligence, at the University of Waterloo.

### Model Specifications
As part of the experiment design, we proposed one full model, which represents our hypothesis, and two other models for comparison. The full model represents a repeated k-means classifier. We choose to use 100 repetitions since this has been shown to effectively reduce the impact of errors due to cluster initialization. We also choose to use [k-means++](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) for cluster initialization to further minimize the chances of bad clustering.

We convert a review into four features: length of review text, days since product posting, agreement between sentiment and product rating, and a novel feature the we called sentiment variability. Length of review text refers to the number of characters in the body text of a review - studies have shown that real reviews have longer body texts. Studies have also shown that more fake reviews are posted closer to the date at which a product was posted, since this is when the fake reviews can have a larger influence on customer decision making. For agreement between sentiment and product rating, we first compute the sentiment of the body text of the review, then find the difference between the scaled sentiment (1 to 5) and the rating. Previous work has shown that real reviews have stronger agreement. Finally, sentiment variability is defined as the variation between the sentiments of sentences in a review. It is expected that real reviews will have higher variability, as real users should be less biased.

The full model uses all four features shown above. Another model, the half model, follows the same methodology but doesn't consider the sentiment variability feature. Finally, the naive model only considers sentiment variability and classifies all reviews above a certain percentile in sentiment variability as real. There are two hyperparameters which are optimized for during the validation stage. The first is what methodology is used to determine sentiment strength. This affects the sentiment variability and agreement between sentiment and product rating features. We consider SentiWordNet, TextBlob and Vader, which are three popular libraries for sentiment analysis. The second is what percentage split of real/fake reviews our model is aiming for, under the training stage. Studies have shown that fake reviews comprise 10-30% of total reviews, so we will iterate throughout this range by 0.5% intervals. During the validation stage, for each model, we consider all commbinations of sentiment methodology and real/fake splits, and choose the best model, of each type, available based on the absolute value of the [Matthews Correlation Coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient).

Finally, during the testing stage, we use the optimal model to classify the reviews in the testing dataset, determining the accuracy, f1 scores, and Matthews Correlation Coefficient of our classification. We can then compare the performance of the three model types. The outputs from our testing can be found in the `/results` folder, so that you can check the results without running the program.

### Datasets
We use three datasets, one for training, validating, and testing the model respectively. Our validation dataset and testing datasets come from the [YelpZip data](http://odds.cs.stonybrook.edu/yelpzip-dataset/) collected by Shebuti Rayana and Leman Akoglu. This has frequently been used in research involving fake reviews and is known as a "gold-standard" dataset. It is split into collections of Yelp Chicago and Yelp NYC data, which make up the validation and testing datasets respectively. They are labelled as real or fake based on the Yelp spam filter.

The training dataset consists of unlabelled [Amazon automotive product data](https://jmcauley.ucsd.edu/data/amazon/) first collected by Julian McAuley.

### Dependencies
The following dependencies are required to run the program:

- VADER: `pip3 install vaderSentiment --user`

- Textblob: `pip3 install -U textblob --user`

- Scikit-Learn: `pip3 install sklearn --user`

- Pandas: `pip3 install pandas`

- Nltk: `pip3 install --user -U nltk`

- Numpy: `pip install --user -U numpy`

Also, follow the instructions [here](http://odds.cs.stonybrook.edu/yelpzip-dataset/) to get access to the YelpZip dataset, which will be used to validate and test the models. Copy the `YelpChi` and `YelpNYC` folders into this project directory before running. The Amazon product dataset is included by default in this repository.

### Running the program
Four programs are provided to run the model: `process_datasets.py`, `test_full_model.py`, `test_half_model.py`, and `test_naive_model.py`. 

The first, `process_datasets.py`, takes in all the data that will be used, and converts it to the feature vectors that are actually used for training, validation, and testing. This is then saved to the `/Data` folder for later use (to train, validate, and test models). Since these datasets are large, this can take multiple hours. After running `python3 process_datasets.py`, the output should look like:
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

The remaining three programs train and validate a full repeated k-means model, a repeated k-means model without considering sentiment variation, and a naive model using only sentiment variation respectively, using the features generated above. They can be called using python, e.g. `python3 test_full_model.py` and will have output similar to:

```
kzzhang@Kevins-MacBook-Pro fake-review-classifier % python3 test_naive_model.py
[nltk_data] Downloading package sentiwordnet to
[nltk_data]     /Users/kzzhang/nltk_data...
[nltk_data]   Package sentiwordnet is already up-to-date!
[nltk_data] Downloading package wordnet to /Users/kzzhang/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
[nltk_data] Downloading package averaged_perceptron_tagger to
[nltk_data]     /Users/kzzhang/nltk_data...
[nltk_data]   Package averaged_perceptron_tagger is already up-to-
[nltk_data]       date!
Training naive model
===================
	loading features
	training models
		Training models for method: VADER
		Training models for method: TextBlob
		Training models for method: SentiWordNet
Validating naive model
===================
	loading features
	validating model parameters
		Validating parameters for method: VADER
			Validation of method: VADER, real percent: 70.0, mcc: 0.08264473285075033
			Validation of method: VADER, real percent: 70.5, mcc: 0.08293145531130643
			Validation of method: VADER, real percent: 71.0, mcc: 0.0827925220403657
			Validation of method: VADER, real percent: 71.5, mcc: 0.08278281972044763
			Validation of method: VADER, real percent: 72.0, mcc: 0.08376549436647668
			Validation of method: VADER, real percent: 72.5, mcc: 0.08521053849464634
			Validation of method: VADER, real percent: 73.0, mcc: 0.08547322606197169
			Validation of method: VADER, real percent: 73.5, mcc: 0.08584639409703046
			Validation of method: VADER, real percent: 74.0, mcc: 0.08582553348677571
			Validation of method: VADER, real percent: 74.5, mcc: 0.08614463071280062
			Validation of method: VADER, real percent: 75.0, mcc: 0.08682463665305573
			Validation of method: VADER, real percent: 75.5, mcc: 0.08789212130177926
			Validation of method: VADER, real percent: 76.0, mcc: 0.08810727140015934
			Validation of method: VADER, real percent: 76.5, mcc: 0.0872942336145041
			Validation of method: VADER, real percent: 77.0, mcc: 0.08765010888560067
			Validation of method: VADER, real percent: 77.5, mcc: 0.0871209474300573
			Validation of method: VADER, real percent: 78.0, mcc: 0.08689139029691391
			Validation of method: VADER, real percent: 78.5, mcc: 0.08771065855562223
			Validation of method: VADER, real percent: 79.0, mcc: 0.08719537203347925
			Validation of method: VADER, real percent: 79.5, mcc: 0.08653513658222997
			Validation of method: VADER, real percent: 80.0, mcc: 0.08597176000987955
			Validation of method: VADER, real percent: 80.5, mcc: 0.08506539354623653
			Validation of method: VADER, real percent: 81.0, mcc: 0.08568993140252033
			Validation of method: VADER, real percent: 81.5, mcc: 0.08603907319204221
			Validation of method: VADER, real percent: 82.0, mcc: 0.08605716908138371
			Validation of method: VADER, real percent: 82.5, mcc: 0.08547243470055492
			Validation of method: VADER, real percent: 83.0, mcc: 0.08509278761022543
			Validation of method: VADER, real percent: 83.5, mcc: 0.08469962527841453
			Validation of method: VADER, real percent: 84.0, mcc: 0.08475103619750514
			Validation of method: VADER, real percent: 84.5, mcc: 0.08426803864889901
			Validation of method: VADER, real percent: 85.0, mcc: 0.08456948897617357
			Validation of method: VADER, real percent: 85.5, mcc: 0.08449176366144835
			Validation of method: VADER, real percent: 86.0, mcc: 0.08436501103174274
			Validation of method: VADER, real percent: 86.5, mcc: 0.08311448820948267
			Validation of method: VADER, real percent: 87.0, mcc: 0.08228072638944878
			Validation of method: VADER, real percent: 87.5, mcc: 0.08314168303161766
			Validation of method: VADER, real percent: 88.0, mcc: 0.08219776270432297
			Validation of method: VADER, real percent: 88.5, mcc: 0.08227797497773495
			Validation of method: VADER, real percent: 89.0, mcc: 0.08234531962265396
			Validation of method: VADER, real percent: 89.5, mcc: 0.08097104509760357
			Validation of method: VADER, real percent: 90.0, mcc: 0.0805550216514432
		Validating parameters for method: TextBlob
			Validation of method: TextBlob, real percent: 70.0, mcc: 0.05881217577165557
			Validation of method: TextBlob, real percent: 70.5, mcc: 0.058788987620996117
			Validation of method: TextBlob, real percent: 71.0, mcc: 0.05988949364801962
			Validation of method: TextBlob, real percent: 71.5, mcc: 0.0598761616435033
			Validation of method: TextBlob, real percent: 72.0, mcc: 0.06150083015064168
			Validation of method: TextBlob, real percent: 72.5, mcc: 0.061326008318612645
			Validation of method: TextBlob, real percent: 73.0, mcc: 0.060680744751702737
			Validation of method: TextBlob, real percent: 73.5, mcc: 0.06188868654927736
			Validation of method: TextBlob, real percent: 74.0, mcc: 0.06185650932574687
			Validation of method: TextBlob, real percent: 74.5, mcc: 0.06158502235535622
			Validation of method: TextBlob, real percent: 75.0, mcc: 0.061759792264798836
			Validation of method: TextBlob, real percent: 75.5, mcc: 0.06359653334361776
			Validation of method: TextBlob, real percent: 76.0, mcc: 0.06351592178436635
			Validation of method: TextBlob, real percent: 76.5, mcc: 0.06388548103769531
			Validation of method: TextBlob, real percent: 77.0, mcc: 0.06554758970023479
			Validation of method: TextBlob, real percent: 77.5, mcc: 0.06634975543256898
			Validation of method: TextBlob, real percent: 78.0, mcc: 0.06606680439936043
			Validation of method: TextBlob, real percent: 78.5, mcc: 0.06637712870631877
			Validation of method: TextBlob, real percent: 79.0, mcc: 0.06681267550230925
			Validation of method: TextBlob, real percent: 79.5, mcc: 0.06696475337645466
			Validation of method: TextBlob, real percent: 80.0, mcc: 0.06727444613132633
			Validation of method: TextBlob, real percent: 80.5, mcc: 0.06778067202771448
			Validation of method: TextBlob, real percent: 81.0, mcc: 0.06766494835996935
			Validation of method: TextBlob, real percent: 81.5, mcc: 0.06924715305568083
			Validation of method: TextBlob, real percent: 82.0, mcc: 0.07050260227550904
			Validation of method: TextBlob, real percent: 82.5, mcc: 0.06949783857232006
			Validation of method: TextBlob, real percent: 83.0, mcc: 0.06990333550424599
			Validation of method: TextBlob, real percent: 83.5, mcc: 0.06834156083131736
			Validation of method: TextBlob, real percent: 84.0, mcc: 0.06847566555658706
			Validation of method: TextBlob, real percent: 84.5, mcc: 0.06916046664710629
			Validation of method: TextBlob, real percent: 85.0, mcc: 0.06829616088539948
			Validation of method: TextBlob, real percent: 85.5, mcc: 0.06936564545651844
			Validation of method: TextBlob, real percent: 86.0, mcc: 0.06954641449926538
			Validation of method: TextBlob, real percent: 86.5, mcc: 0.0704107766473018
			Validation of method: TextBlob, real percent: 87.0, mcc: 0.07107658739039718
			Validation of method: TextBlob, real percent: 87.5, mcc: 0.07171664623454406
			Validation of method: TextBlob, real percent: 88.0, mcc: 0.07188832459547695
			Validation of method: TextBlob, real percent: 88.5, mcc: 0.07189798774360019
			Validation of method: TextBlob, real percent: 89.0, mcc: 0.07102408900286077
			Validation of method: TextBlob, real percent: 89.5, mcc: 0.06948665124887972
			Validation of method: TextBlob, real percent: 90.0, mcc: 0.07063526593681521
		Validating parameters for method: SentiWordNet
			Validation of method: SentiWordNet, real percent: 70.0, mcc: 0.0640693395670138
			Validation of method: SentiWordNet, real percent: 70.5, mcc: 0.06529180015884453
			Validation of method: SentiWordNet, real percent: 71.0, mcc: 0.06649108538754589
			Validation of method: SentiWordNet, real percent: 71.5, mcc: 0.0663704556730222
			Validation of method: SentiWordNet, real percent: 72.0, mcc: 0.06616370902501202
			Validation of method: SentiWordNet, real percent: 72.5, mcc: 0.06698158814968787
			Validation of method: SentiWordNet, real percent: 73.0, mcc: 0.06586292903203494
			Validation of method: SentiWordNet, real percent: 73.5, mcc: 0.06671574990460914
			Validation of method: SentiWordNet, real percent: 74.0, mcc: 0.06678545343553685
			Validation of method: SentiWordNet, real percent: 74.5, mcc: 0.06762119243227681
			Validation of method: SentiWordNet, real percent: 75.0, mcc: 0.06806850460095508
			Validation of method: SentiWordNet, real percent: 75.5, mcc: 0.0697164681021902
			Validation of method: SentiWordNet, real percent: 76.0, mcc: 0.07203807413466885
			Validation of method: SentiWordNet, real percent: 76.5, mcc: 0.07110288299419433
			Validation of method: SentiWordNet, real percent: 77.0, mcc: 0.0716417611027003
			Validation of method: SentiWordNet, real percent: 77.5, mcc: 0.0719349727671624
			Validation of method: SentiWordNet, real percent: 78.0, mcc: 0.07278207550826586
			Validation of method: SentiWordNet, real percent: 78.5, mcc: 0.07287378318061227
			Validation of method: SentiWordNet, real percent: 79.0, mcc: 0.07380494983614051
			Validation of method: SentiWordNet, real percent: 79.5, mcc: 0.07328866822617144
			Validation of method: SentiWordNet, real percent: 80.0, mcc: 0.07434716525698458
			Validation of method: SentiWordNet, real percent: 80.5, mcc: 0.07425469375564123
			Validation of method: SentiWordNet, real percent: 81.0, mcc: 0.07432226801063788
			Validation of method: SentiWordNet, real percent: 81.5, mcc: 0.07525116150268094
			Validation of method: SentiWordNet, real percent: 82.0, mcc: 0.07686037137514824
			Validation of method: SentiWordNet, real percent: 82.5, mcc: 0.0773129918721924
			Validation of method: SentiWordNet, real percent: 83.0, mcc: 0.07655131255852794
			Validation of method: SentiWordNet, real percent: 83.5, mcc: 0.07750018455611131
			Validation of method: SentiWordNet, real percent: 84.0, mcc: 0.0771877833771335
			Validation of method: SentiWordNet, real percent: 84.5, mcc: 0.07669070122261554
			Validation of method: SentiWordNet, real percent: 85.0, mcc: 0.07462126018416007
			Validation of method: SentiWordNet, real percent: 85.5, mcc: 0.07479149847740324
			Validation of method: SentiWordNet, real percent: 86.0, mcc: 0.07574156416404162
			Validation of method: SentiWordNet, real percent: 86.5, mcc: 0.07713980139037013
			Validation of method: SentiWordNet, real percent: 87.0, mcc: 0.07569488260070699
			Validation of method: SentiWordNet, real percent: 87.5, mcc: 0.07538340163104253
			Validation of method: SentiWordNet, real percent: 88.0, mcc: 0.0741365002595952
			Validation of method: SentiWordNet, real percent: 88.5, mcc: 0.07388881341408723
			Validation of method: SentiWordNet, real percent: 89.0, mcc: 0.07343581510560565
			Validation of method: SentiWordNet, real percent: 89.5, mcc: 0.07414243567327257
			Validation of method: SentiWordNet, real percent: 90.0, mcc: 0.07396993030681631
		Choosing best method
			VALIDATION: MCC using VADER and real percent 76.0 is 0.08810727140015934
			VALIDATION: MCC using TextBlob and real percent 76.0 is 0.07189798774360019
			VALIDATION: MCC using SentiWordNet and real percent 76.0 is 0.07750018455611131
Best model uses: VADER and real percentage: 76.0
Testing naive model
===================
	loading features
	testing using test dataset
Test model accuracy:0.8213629223622205, f1score: 0.8997568157238328, mcc: 0.081028922192941
Total execution time: 0:00:17.856245
```
