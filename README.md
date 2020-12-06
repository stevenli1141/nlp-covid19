# CS451 final project

Use Twitter data (tweets) to predict case numbers for COVID-19.

## Data sources

The daily COVID-19 cases in the United States was obtained here:
https://www.kaggle.com/sudalairajkumar/covid19-in-usa

COVID-19 twitter data from July to August is from here:
https://www.kaggle.com/gpreda/covid19-tweets

COVID-19 twitter processed n-gram data from January 23 to November 1 is from here:
https://www.kaggle.com/paulrohan2020/covid19-unique-tweets

## Goal

Given twitter text data and number of new positive cases on a certain day, we want to forecast the number of new positive cases the next day. New cases are related to the previous day but may increase or decrease based on conditions on the ground e.g. lockdowns, quarantines, mask wearing. Tweets are a noisy signal of the true state of the world which we can leverage to estimate the number of cases tomorrow based on today.


