import pyspark
from pyspark.sql import SparkSession

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re

from functools import partial
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# process unigram data

def process_row(features, row):
  word = row.__getitem__('word1')
  freq = int(row.__getitem__('n'))
  date = row.__getitem__('date')
  if word not in features:
    return
  yield (date.split('T')[0].replace('-', ''), np.array([freq if w == word else 0 for w in features]))

def build_word_frequency_by_date(tweets, features):
  t = tweets.rdd.flatMap(partial(process_row, features)).reduceByKey(lambda l, r: l+r)
  t = t.filter(lambda r: r[0] >= '20200301') # filter to when cases started rising
  return t

# Linear regression model

K = 4 # set number of partitions to use when building model

def to_freq_vector(words, features):
  # limit to a smaller number of words to prevent overfitting
  return [words.get(w, 0) for w in features]

def build_model(stream):
  # stream: a generator of (date, (bag of words, label))
  # Load word index from pickle (run wordcount first)
  data = list(stream)
  x = np.array(map(lambda row: row[0], data)) # features
  y = np.array(map(lambda row: row[1], data)) # labels
  if len(x) == 0:
    return

  print('Partition dataset size %d' % len(y))
  # we can change which model to use here
  # reg = LinearRegression().fit(x, y)
  reg = RandomForestRegressor(50).fit(x, y)
  yield reg

# Model evaluators

def produce_scatterplot(x, y, xlab, ylab, filename, line=False):
  plt.clf()
  plt.scatter(x, y)
  if line:
    t = [np.min(x), np.max(x)]
    plt.plot(t, t, '--')
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.savefig(filename)

def produce_lineplot(y1, y2, lab1, lab2, xlab, ylab, filename):
  plt.clf()
  plt.plot(y1, '-', label=lab1)
  plt.plot(y2, '-', label=lab2)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.legend()
  plt.savefig(filename)

def produce_hist(x, title, xlab, ylab, filename):
  plt.clf()
  plt.hist(x, rwidth=0.9)
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  plt.title(title)
  plt.savefig(filename)

if __name__ == '__main__':
  spark = SparkSession.builder.getOrCreate()
  spark.conf.set('spark.sql.execution.arrow.enabled', 'true')

  tweets = spark.read.option('multiline', True).option('escape', '"').csv('covid19tweets/*csv', header=True)

  features = pickle.load(open('selected.pickle', 'r'))
  x = build_word_frequency_by_date(tweets, features)

  # case numbers can easily fit into one pandas dataframe
  cases_pd = pd.read_csv('us_covid19_daily.csv')
  cases_pd['nextday'] = cases_pd['positiveIncrease'].shift(1) # get previous day numbers
  cases = spark.createDataFrame(cases_pd[['date', 'positiveIncrease', 'nextday']].dropna())
  y = cases.dropna().rdd.map(lambda r: (str(r[0]), (int(r[1]), int(r[2]))))

  # join and format to (date, (words, positive, nextday))
  df = x.join(y).map(lambda r: (r[0], (r[1][0], r[1][1][0], r[1][1][1])))
  # result = df.collect()
  # for row in result:
  #   print(row)

  # process into word frequency vectors and append current no. of cases to feature vector
  df = df.map(lambda r: (r[0], (np.append(r[1][0], r[1][1]), r[1][2])))

  dates = df.map(lambda r: r[0]).sortBy(lambda r: r).collect()

  # train test split by date then remove date from labels (not needed for our purposes)
  split = dates[int(len(dates) * 0.8)] # gets us an approximate 4:1 train:test split
  train_df = df.filter(lambda r: r[0] < split).map(lambda r: r[1])
  test_df = df.filter(lambda r: r[0] >= split).sortByKey().map(lambda r: r[1])
  xtest = np.array(test_df.map(lambda r: r[0]).collect())
  ytest = np.array(test_df.map(lambda r: r[1]).collect())

  train_size = train_df.count()
  test_size = test_df.count()

  # fit models
  models = train_df.repartition(K).mapPartitions(build_model).collect()

  # get ensemble prediction results
  print('Train %d, test %d, %d percent split' % (train_size, test_size, 100 * train_size / (train_size + test_size)))
  aggr = np.array([model.predict(xtest) for model in models])
  ypred = np.mean(aggr, axis=0)
  print('Predicted:')
  print(ypred)
  print('Actual:')
  print(ytest)
  print('Diff:')
  print(ypred - ytest)
  print('Ratio:')
  print(ypred / ytest)
  print('Random forest ensemble RMSE: %f' % np.sqrt(mean_squared_error(ypred, ytest)))

  # validation of general case
  print('Average test y: %f', np.mean(ytest))
  print('SD test y: %f', np.std(ytest))

  # Plot predicted next day cases against current day cases to check if we are actually doing anything useful
  produce_scatterplot(xtest.take(indices=-1, axis=1), ytest,
    xlab='Current day cases', ylab='Real next day cases', filename='rf_scatter.png')
  produce_scatterplot(xtest.take(indices=-1, axis=1), ypred,
    xlab='Current day cases', ylab='Predicted next day cases', filename='rf_corr.png')
  produce_scatterplot(ytest, ypred, xlab='Actual day cases', ylab='Predicted day cases', filename='rf_pred.png', line=True)
  produce_hist(ypred / ytest, 'Predicted to Actual Ratios', xlab='Predicted to Actual case count ratio', ylab='Frequency', filename='rf_ratio.png')
  produce_lineplot(ytest, ypred, 'Actual', 'Predicted', xlab='Day', ylab='Number of Cases', filename='rf_trend.png')
