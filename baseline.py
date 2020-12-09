import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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

def produce_lineplot(y1, y2, lab1, lab2, filename):
  plt.clf()
  plt.plot(y1, '-', label=lab1)
  plt.plot(y2, '-', label=lab2)
  plt.legend()
  plt.savefig(filename)

def produce_hist(x, title, filename):
  plt.clf()
  plt.hist(x, rwidth=0.9)
  plt.title(title)
  plt.savefig(filename)

if __name__ == '__main__':
  cases_pd = pd.read_csv('us_covid19_daily.csv')
  cases_pd['nextday'] = cases_pd['positiveIncrease'].shift(1) # get previous day numbers
  df = cases_pd[['date', 'positiveIncrease', 'nextday']].dropna().sort_values(by='date', ascending=True)
  df = df.loc[df['date'] >= 20200301] # filter to Mar 1 and after when cases are significant

  dates = sorted(df['date'])

  # train test split by date then remove date from labels (not needed for our purposes)
  split = dates[int(len(dates) * 0.8)] # gets us an approximate 4:1 train:test split

  train_df = df.loc[df['date'] < split, ['positiveIncrease', 'nextday']]
  test_df = df.loc[df['date'] >= split, ['positiveIncrease', 'nextday']]
  xtrain = np.array(train_df['positiveIncrease']).reshape(-1, 1)
  ytrain = np.array(train_df['nextday'])
  xtest = np.array(test_df['positiveIncrease']).reshape(-1, 1)
  ytest = np.array(test_df['nextday'])

  train_size = train_df.shape[0]
  test_size = test_df.shape[0]

  # fit a simple 1d linear model
  model = LinearRegression().fit(xtrain, ytrain)

  # get ensemble prediction results
  print('Train %d, test %d, %d percent split' % (train_size, test_size, 100 * train_size / (train_size + test_size)))
  ypred = model.predict(xtest)
  print('Ensemble aggregation results')
  print('Predicted:')
  print(ypred)
  print('Actual:')
  print(ytest)
  print('Diff:')
  print(ypred - ytest)
  print('Ratio:')
  print(ypred / ytest)
  print('AR(1) model RMSE: %f' % np.sqrt(mean_squared_error(ypred, ytest)))

  # validation of general case
  print('Average test y: %f', np.mean(ytest))
  print('SD test y: %f', np.std(ytest))

  # Plot predicted next day cases against current day cases to check if we are actually doing anything useful
  produce_scatterplot(xtest.take(indices=-1, axis=1), ytest,
    xlab='Current day cases', ylab='Real next day cases', filename='ar1_scatter.png')
  produce_scatterplot(xtest.take(indices=-1, axis=1), ypred,
    xlab='Current day cases', ylab='Predicted next day cases', filename='ar1_corr.png')
  produce_scatterplot(ytest, ypred, 'Actual', 'Predicted', filename='ar1_pred.png', line=True)
  produce_hist(ypred / ytest, 'Predicted to Actual Ratios', 'ar1_ratio.png')
  produce_lineplot(ytest, ypred, 'Actual', 'Predicted', 'ar1_trend.png')
