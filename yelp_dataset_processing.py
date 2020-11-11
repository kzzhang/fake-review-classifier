import pandas as pd

# Parse reviews and metadata in .txt files
def parse_txt_review(filepath, is_metadata, NYC_dataset=False):
  reviews = []
  with open(filepath) as infile:
    for review in infile:
      if is_metadata:
        line = review.split()
        reviews.append(line)
      else:
        if NYC_dataset:
          review_text = review.split('\t')[-1]
          reviews.append(review_text)
        else:
          reviews.append(review)
  
  infile.close()

  return reviews

def parse_yelpChi_dataset(metadata_filepath, review_filepath):
  # Gather reviews and metadata
  reviews_metadata = parse_txt_review(metadata_filepath, True)
  reviews = parse_txt_review(review_filepath, False)

  # Create dataframe
  column_names = [
    'Date',
    'reviewID',
    'reviewerID',
    'productID',
    'label',
    'misc1',
    'misc2',
    'misc3',
    'rating'
  ]

  df = pd.DataFrame(reviews_metadata, columns=column_names)
  df['reviewText'] = reviews

  return df

def parse_yelpNYC_dataset(metadata_filepath, review_filepath):
  # Gather reviews and metadata
  reviews_metadata = parse_txt_review(metadata_filepath, True)
  reviews = parse_txt_review(review_filepath, False, True)

  # Create dataframe
  column_names = [
    'reviewerID',
    'productID',
    'rating',
    'label',
    'Date'
  ]

  df = pd.DataFrame(reviews_metadata, columns=column_names)
  df['reviewText'] = reviews

  return df

# Cleans up Yelp datasets and only maintains necessary data fields
def modify_yelp_dataset(df, columns_to_remove):
  # Drop any row with an empty field
  df.dropna(axis=0, how='any', inplace=True)

  # Calculate time since first review
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.groupby('productID').apply(lambda x: x.sort_values('Date'))
  df['daysSinceFirstReview'] = df['Date'] - df.groupby(level='productID')['Date'].transform('first')
  df['daysSinceFirstReview'] = df['daysSinceFirstReview'].dt.days

  # Determine initial length of review text and assign ID
  df['reviewTextLength'] = df['reviewText'].str.len()
  if 'reviewID' not in df.columns:
    df.insert(0, 'reviewID', range(len(df)))

  # Drop unnecessary columns
  for column in columns_to_remove:
    df.drop(column, axis=1, inplace=True)

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
