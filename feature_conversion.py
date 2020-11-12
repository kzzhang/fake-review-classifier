import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from swn_sentiment import wordnet_senti


INPUT_REVIEW_ID_COL = 0
INPUT_REVIEW_TEXT_COL = 2
INPUT_REVIEW_LENGTH_COL = 3
INPUT_REVIEW_RATING_COL = 4
INPUT_REVIEW_DAYS_FROM_FIRST_COL = 5

def compute_values(sentence, sentiment_analysis_method):
    if sentiment_analysis_method=='VADER':
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(sentence)
        sentence_rating = 2.5 + 2.5*score['pos'] - 2.5*score['neg']
        return score['compound'], sentence_rating
    if sentiment_analysis_method=='TextBlob':
        blob = TextBlob(sentence)
        sentiment = blob.sentiment
        sentence_rating = 2.5+2.5*sentiment.polarity
        return sentiment.polarity,sentence_rating
    if sentiment_analysis_method=='SentiWordNet':
        pos_sentiment, neg_sentiment = wordnet_senti(sentence)
        sentence_rating = 2.5 + 2.5*pos_sentiment - 2.5*neg_sentiment
        compound_score = pos_sentiment - neg_sentiment
        return compound_score, sentence_rating

def calculate_sentiment_vals(paragraph, rating, sentiment_analysis_method):
    sentence_compound_sentiments = []
    sentence_ratings = []
    for sentence in paragraph.split("."):
        compound_score, sentence_rating = compute_values(sentence, sentiment_analysis_method)
        sentence_compound_sentiments.append(compound_score)
        sentence_ratings.append(sentence_rating)

    rating_agreement = abs(rating-np.mean(sentence_ratings))
    sentiment_variability = np.var(sentence_compound_sentiments)
    return rating_agreement,sentiment_variability

# Take in a 2-d vector (data) where each row corresponds to a review
#   Input Cols: review id, product id, review text (pre-processed), pre-processed review length, rating, days from release
# Outputs a new 2-d vector where each row is a review, but all numeric quantities (non-normalized)
#   Output Cols: Review Id, Review length, Rating/sentiment agreement, Review date, sentiment variability
def convert_to_features(data, sentiment_analysis_method):
    output = []
    for review in data:
        rating_agreement,sentiment_variability = calculate_sentiment_vals(
            review[2],float(review[4]),sentiment_analysis_method)
        output.append(np.array([\
                                review[3], # review length
                                rating_agreement, #rating/sentiment agreement
                                review[5], #days since first review of product
                                sentiment_variability, #sentiment variability
                                ]))
    
    return np.asarray(output)

