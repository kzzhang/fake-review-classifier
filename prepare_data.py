import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import yelp_dataset_processing as ydp
import amazon_dataset_processing as adp

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Returns the Amazon dataset used to train the model
def get_training_dataset():
  df = adp.parse_amazon_dataset('amazon_automotive_reviews.json')
  df = adp.modify_amazon_dataset(df)

  return df

# Returns the YelpChi dataset used for validation
def get_validation_dataset():
  hotel_reviews = 'YelpChi/output_review_yelpHotelData_NRYRcleaned.txt'
  hotel_metadata = 'YelpChi/output_meta_yelpHotelData_NRYRcleaned.txt'
  restaurant_reviews = 'YelpChi/output_review_yelpResData_NRYRcleaned.txt'
  restaurant_metadata = 'YelpChi/output_meta_yelpResData_NRYRcleaned.txt'

  # Parse the hotel and restaurants datasets
  hotels_df = ydp.parse_yelpChi_dataset(hotel_metadata, hotel_reviews)
  restaurants_df = ydp.parse_yelpChi_dataset(restaurant_metadata, restaurant_reviews)

  # Modify datasets and remove unnecessary columns
  columns_to_remove = [
    'Date',
    'reviewerID',
    'misc1',
    'misc2',
    'misc3'
  ]
  hotels_df = ydp.modify_yelp_dataset(hotels_df, columns_to_remove)
  restaurants_df = ydp.modify_yelp_dataset(restaurants_df, columns_to_remove)

  df = pd.concat([hotels_df, restaurants_df])

  return df

# Returns the YelpNYC dataset used for testing the model
def get_testing_dataset():
  review_filepath = 'YelpNYC/reviewContent'
  metadata_filepath = 'YelpNYC/metadata'
  df = ydp.parse_yelpNYC_dataset(metadata_filepath, review_filepath)

  # Modify dataset and remove unnecessary columns
  columns_to_remove = [
    'reviewerID',
    'Date'
  ]
  df = ydp.modify_yelp_dataset(df, columns_to_remove)

  return df

# Removes stopwords in provided text, where text is tokenized
def filter_stopwords(tokenized_text):
  filtered = []
  stop_words = stopwords.words("english")

  for word in tokenized_text:
    if word not in stop_words:
      filtered.append(word)

  return filtered

# Removes words that include digits in provided text, where text is tokenized
def remove_noise(tokenized_text):
  filtered = []
  
  for word in tokenized_text:
    contains_digit = any(map(str.isdigit, word))
    if not contains_digit:
      filtered.append(word)
  
  return filtered

# Apply stemming to provided text, where text is tokenized
def stem(tokenized_text):
  stemmed = []
  stemmer = SnowballStemmer("english")

  for word in tokenized_text:
    stemmed.append(stemmer.stem(word))

  return stemmed

# Apply lemmatization to provided text, where text is tokenized
def lemmatize(tokenized_text):
  lemmatized = []
  lem = WordNetLemmatizer()

  for word in tokenized_text:
    lemmatized.append(lem.lemmatize(word))

  return lemmatized

# Preprocesses review text based on sentiment analysis method
#   full_processing = True - this is used for SentiWordNet
#   full_processing = False (default) - this is used for VADER and TextBlob
#   Returns 2d numpy array containing processed dataset
def preprocess(df, full_processing=False):
  # Tokenize review text
  df['reviewText'] = df['reviewText'].apply(word_tokenize)

  # Remove noise
  df['reviewText'] = df['reviewText'].apply(remove_noise)

  # Perform full processing, if needed
  if full_processing:
    df['reviewText'] = df['reviewText'].apply(filter_stopwords)
    df['reviewText'] = df['reviewText'].apply(lemmatize)
  
  # Detokenize text
  df['reviewText'] = df['reviewText'].apply(TreebankWordDetokenizer().detokenize)

  return df.to_numpy()
