# Prediction-of-Netflix-Stock-Price
Predicting Netflix Stock Price with Linear Regression Algorithm. Special thanks to Dr. Mohammed Al-Obaydee for the guidance.

## Business Understanding

Case Study: To design a stock prediction model for the closing price of Netflix.

* Goal: To predict the closing price of Netflix.
* Objective: To build a predictive model using Linear Regression algorithm.

*Closing price: Last price to buy before the closing of business hour or After-Hour Trading (AHT).*


## Analytic Approach

Linear Regression algorithm is used to predict the closing price of Netflix stock.

*Simple Linear Regression algorithm is a supervised machine learning algorithm to model linear relationship between to variables.*

## Data Requirements

* Opening Price
* Closing Price
* Low Price
* High Price
* Volume of Stock

## Data Collection

The dataset is sourced from Kaggle.com. Dataset is downloaded and uploaded back to Jupyter Notebook. This dataset is based on Yahoo Finance: Netflix historical price 12/16/2015 until 12/16/2019 daily price and volume. There are 7 columns; Date, open, high, low, close, volume, adj close (2001, 7) each of stock.

## Data Understanding and Preparation

* Importing Packages

```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
```

* Reading the Dataset

```
df = pd.read_csv("NFLX.csv")
df.head()
```

```
df.tail()
```

```
df.shape
```

(1007 rows, 7 columns)

* Summary Statistics

```
df.describe()
```

```
df.describe(include="all")
```
*'all' : All columns of the input will be included in the output.*

* Finding Correlation

To find the correlation among the columns in the dataframe using ‘Pearson’ method.

```
df.corr(method="pearson")
```

```
corr = df.corr()
corr
```

* Exploratory Data Analysis (EDA)

Using heatmap for visualisation.

```
import seaborn as sb
sb.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns,
 cmap='RdBu_r', annot=True, linewidth=0.5)
 ```
 ![1](https://user-images.githubusercontent.com/93753467/149455943-b2d33f1f-29aa-4ebd-aa7f-568fdd2e7446.png)

 
 

