import pyspark
from pyspark.sql import SparkSession

from nltk.corpus import stopwords
import re

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


# ML

def build_model(data):
  # data: a generator of (date, (bag of words, label))
  # Idea: do a wordcount elsewhere and then decide on which words should be used in feature vector

if __name__ == '__main__':
  spark = SparkSession.builder.getOrCreate()

  tweets = spark.read.option('multiline', True).option('escape', '"').csv('covid19_tweets.csv', header=True)
  tweets = tweets.sample(False, 0.05) # sample 5% of total rows

  x = build_word_frequency_by_date(tweets.select(['date', 'text']).dropna())

  cases = spark.read.csv('us_covid19_daily.csv', header=True)
  y = cases.select(['date', 'positiveIncrease']).dropna().rdd.map(lambda r: (r[0], int(r[1])))

  df = x.join(y)
  result = df.collect()
  for row in result:
    print(row)

  # df.mapPartitions(build_model)
