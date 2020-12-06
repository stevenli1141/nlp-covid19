import pyspark
from pyspark.sql import SparkSession

from nltk.corpus import stopwords
import pickle
import re

stopwords_en = stopwords.words('english')

N = 2000 # top N words

def wordcount(row):
  text = row.__getitem__('text')
  words = map(lambda t: re.sub('^([^A-Za-z0-9])+|([^A-Za-z0-9])+$', '', t.lower()), text.split(' '))
  for w in words:
    # Use words with length of at least 3
    if len(w) > 2 and w not in stopwords_en:
      yield (w, 1)

if __name__ == '__main__':
  spark = SparkSession.builder.getOrCreate()

  tweets = spark.read.option('multiline', True).option('escape', '"').csv('covid19_tweets.csv', header=True)

  x = tweets.select('text').dropna()
  result = x.rdd.flatMap(wordcount).reduceByKey(lambda l, r: l+r).takeOrdered(N, key=lambda r: -r[1])
  for r in result:
    print((r[1], r[0]))

  words = map(lambda r: r[0], result)
  with open('topwords.pickle', 'w') as f:
    pickle.dump(words, f)
