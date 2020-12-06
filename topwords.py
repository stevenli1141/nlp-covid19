import pyspark
from pyspark.sql import SparkSession

from nltk.corpus import stopwords
import pickle
import re

stopwords_en = stopwords.words('english')

N = 5000 # top N words

if __name__ == '__main__':
  spark = SparkSession.builder.getOrCreate()

  df = spark.read.option('multiline', True).option('escape', '"').csv('covid19tweets/*.csv', header=True)
  result = df.dropna().rdd.map(lambda r: (r[0], int(r[1]))).reduceByKey(lambda l, r: l+r).takeOrdered(N, key=lambda r: -r[1])
  for r in result:
    print((r[1], r[0]))
  words = map(lambda r: r[0], result)
  with open('topwords.pickle', 'w') as f:
    pickle.dump(words, f)
