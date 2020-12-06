import pyspark
from pyspark.sql import SparkSession

import numpy as np
import pandas as pd
import pickle
import re

from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# process text data

stopwords_en = stopwords.words('english')

def get_word_frequency(text):
  words = map(lambda t: re.sub('^([^A-Za-z0-9])+|([^A-Za-z0-9])+$', '', t.lower()), text.split(' '))
  freq = {}
  for w in words:
    if len(w) > 1 and w not in stopwords_en:
      freq[w] = freq.get(w, 0) + 1
  return freq

def add_freq(a, b):
  freq = {}
  for k, v in a.items():
    freq[k] = v
  for k, v in b.items():
    freq[k] = freq.get(k, 0) + v
  return freq

def process_row(row):
  date = row.__getitem__('date')
  text = row.__getitem__('text')
  return (date.split(' ')[0].replace('-', ''), get_word_frequency(text))

def remove_low_freq_words(row):
  freq = row[1]
  return (row[0], {k:v for k, v in freq.items() if v >= 5})

def build_word_frequency_by_date(tweets):
  return tweets.rdd.map(process_row).reduceByKey(add_freq).map(remove_low_freq_words).sortByKey()


# Linear regression model

K = 4 # number of partitions in ensemble. should be large enough to scale
M = 50 # limit to D features to prevent overfitting

def to_freq_vector(words, features):
  # limit to a smaller number of words to prevent overfitting
  return [words.get(w, 0) for w in features[:M]]

def build_model(stream):
  # stream: a generator of (date, (bag of words, label))
  # Load word index from pickle (run wordcount first)
  data = list(stream)
  x = np.array(map(lambda row: row[0], data)) # features
  y = np.array(map(lambda row: row[1], data)) # labels
  if len(x) == 0:
    return

  # we can change which model to use here
  reg = LinearRegression().fit(x, y)
  yield reg

if __name__ == '__main__':
  spark = SparkSession.builder.getOrCreate()
  spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

  tweets = spark.read.option('multiline', True).option('escape', '"').csv('covid19_tweets.csv', header=True)
  tweets = tweets.sample(False, 0.2) # sample 20% of total rows to save time for now

  x = build_word_frequency_by_date(tweets.select(['date', 'text']).dropna())

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
  features = pickle.load(open('selected.pickle', 'r'))
  # df = df.map(lambda r: (r[0], (to_freq_vector(r[1][0], features) + [r[1][1]], r[1][2])))
  df = df.map(lambda r: (r[0], (to_freq_vector(r[1][0], features), r[1][2])))

  # train test split by date then remove date from labels (not needed for our purposes)
  SPLIT = '20200817' # gets us an approximate 4:1 train:test split
  train_df = df.filter(lambda r: r[0] < SPLIT).map(lambda r: r[1])
  test_df = df.filter(lambda r: r[0] >= SPLIT).map(lambda r: r[1])
  xtest = np.array(test_df.map(lambda r: r[0]).collect())
  ytest = np.array(test_df.map(lambda r: r[1]).collect())

  train_size = train_df.count()
  test_size = test_df.count()

  # fit models
  models = train_df.repartition(K).mapPartitions(build_model).collect()

  # get bagging prediction results
  print('Train %d, test %d, %d percent split' % (train_size, test_size, 100 * train_size / (train_size + test_size)))
  bagging = np.array([model.predict(xtest) for model in models])
  ypred = np.mean(bagging, axis=0)
  print('Bootstrap aggregation (bagging) results')
  print('Test X:')
  print(xtest)
  print('Predicted:')
  print(ypred)
  print('Actual:')
  print(ytest)
  print('Ratio:')
  print(ypred / ytest)
  print('Bagging RMSE: %f' % np.sqrt(mean_squared_error(ypred, ytest)))

  # validation of general case
  ytrain = train_df.map(lambda r: r[1]).collect()
  y_all = np.array([y for y in ytest] + ytrain)
  print(y_all)
  print('Average case count: %f', np.mean(y_all))
  print('SD case count: %f', np.std(y_all))
