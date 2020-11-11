import pandas as pd

# Parse various JSON reviews using a generator object
def parse_json(filepath):
  with open(filepath) as infile:
    for review in infile:
      yield eval(review)
  
  infile.close()

# Create a dataframe containing Amazon reviews
def parse_amazon_dataset(filepath):
  review_index = 0
  reviews = {}

  for review in parse_json(filepath):
    reviews[review_index] = review
    review_index += 1

  return pd.DataFrame.from_dict(reviews, orient='index')

# Cleans up dataset and only maintains necessary data fields
def modify_amazon_dataset(df):
  # Drop any row with an empty field
  df.dropna(axis=0, how='any', inplace=True)

  # Calculate time since first review
  df['reviewTime'] = pd.to_datetime(df['reviewTime'])
  df = df.groupby('asin').apply(lambda x: x.sort_values('unixReviewTime'))
  df['daysSinceFirstReview'] = df['reviewTime'] - df.groupby(level = 'asin')['reviewTime'].transform('first')
  df['daysSinceFirstReview'] = df['daysSinceFirstReview'].dt.days

  # Determine initial length of review text and assign ID
  df['reviewTextLength'] = df['reviewText'].str.len()
  df.insert(0, 'reviewID', range(len(df)))

  # Drop unnecessary columns
  df.drop('reviewerID', axis=1, inplace=True)
  df.drop('reviewerName', axis=1, inplace=True)
  df.drop('helpful', axis=1, inplace=True)
  df.drop('summary', axis=1, inplace=True)
  df.drop('unixReviewTime', axis=1, inplace=True)
  df.drop('reviewTime', axis=1, inplace=True)

  # Rename certain columns
  renamed_columns = {
    'asin'      :   'productID',
    'overall'   :   'rating'
  }
  df.rename(columns=renamed_columns, inplace=True)

  # Reset index
  df.reset_index()
  df.index = range(len(df))

  # Reorder columns
  column_order = [
    'reviewID',
    'productID',
    'reviewText',
    'reviewTextLength',
    'rating',
    'daysSinceFirstReview'
  ]
  df = df[column_order]
  
  return df
