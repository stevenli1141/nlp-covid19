import pyspark
from pyspark.sql import SparkSession

import numpy as np
import pandas as pd
import pickle
import re

from nltk.corpus import stopwords
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso

# process text data

stopwords_en = stopwords.words('english')

def add_freq(a, b):
  freq = {}
  for k, v in a.items():
    freq[k] = v
  for k, v in b.items():
    freq[k] = freq.get(k, 0) + v
  return freq

def process_row(row):
  word = row.__getitem__('word1')
  freq = row.__getitem__('n')
  date = row.__getitem__('date')
  return (date.split('T')[0].replace('-', ''), { word: int(freq) })

def remove_low_freq_words(row):
  freq = row[1]
  return (row[0], {k:v for k, v in freq.items() if v >= 5})

def build_word_frequency_by_date(tweets):
  return tweets.rdd.map(process_row).reduceByKey(add_freq).map(remove_low_freq_words).sortByKey()


# Select features using F-regression
# As a simple naive approach, take top N words by F-score

N = 100 # top N words by F-score to use for features

def to_freq_vector(words, topwords):
  # limit to a smaller number of words to prevent overfitting
  return [words.get(w, 0) for w in topwords]

def get_scores(x, y, topwords):
  scores = f_regression(x, y)
  wordscores = sorted([
    (scores[0][i], scores[1][i], w) for i, w in enumerate(topwords) if not np.isnan(scores[0][i])
  ], reverse=True)
  print('f-value, p-value, word')
  for score in wordscores:
    print(score)
  print('f-value of current day case count: %f' % scores[0][-1])
  print('p-value of current day case count: %f' % scores[1][-1])
  topscores = [score[2] for score in wordscores[:N]]
  with open('selected.pickle', 'w') as f:
    pickle.dump(topscores, f)

if __name__ == '__main__':
  spark = SparkSession.builder.getOrCreate()
  spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

  tweets = spark.read.option('multiline', True).option('escape', '"').csv('covid19tweets/*.csv', header=True)
  tweets = tweets.sample(False, 0.05) # sample 20% of total rows to save time for now

  x = build_word_frequency_by_date(tweets)
  print(x)

  # case numbers can easily fit into one pandas dataframe
  cases_pd = pd.read_csv('us_covid19_daily.csv')
  cases_pd['nextday'] = cases_pd['positiveIncrease'].shift(1) # get previous day numbers
  cases = spark.createDataFrame(cases_pd[['date', 'positiveIncrease', 'nextday']].dropna())
  y = cases.dropna().rdd.map(lambda r: (str(r[0]), (int(r[1]), int(r[2]))))

  # join and format to (date, (words, positive, nextday))
  df = x.join(y).map(lambda r: (r[0], (r[1][0], r[1][1][0], r[1][1][1])))
  result = df.collect()
  # for row in result:
  #   print(row)

  # process into word frequency vectors and append current no. of cases to feature vector
  f = open('topwords.pickle', 'r')
  topwords = pickle.load(f)
  df = df.map(lambda r: (to_freq_vector(r[1][0], topwords) + [r[1][1]], r[1][2]))

  data = df.collect()
  nx = np.array(map(lambda row: row[0], data)) # features
  ny = np.array(map(lambda row: row[1], data)) # labels
  get_scores(nx, ny, topwords)

