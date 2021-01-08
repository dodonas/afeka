# Introduction to Intelligent systems
# Final project
> In this project, we will perform preliminary processing of the data set and then the selected data processing, and finally post-processing the results. 
The project will contain all the necessary comments and explanations. The data will also be visible for better understanding.
In order for it to run non-stop for presentation purposes only, we will assume that some manual action was taken where expected (i.e., data processing and decision making)

# Submitters:
[Andrey Dodon](https://www.kaggle.com/andreydodon) - `Afeka M.Sc. student, Intelligent systems`

[Michael Gudovsky](https://il.linkedin.com/in/michael-gudovsky-1392157b) - `Afeka M.Sc. student, Intelligent systems`


# Resources:
  - [Kaggle](https://www.kaggle.com/antfarol/car-sale-advertisements/download) - Anton Bobanov's dataset
  - Python libraries:
    * [pandas](https://pandas.pydata.org/)
    * [scikit-learn](https://scikit-learn.org/stable/)
    * [Statsmodel](https://www.statsmodels.org/stable/index.html)
    * [Matplotlib](https://matplotlib.org/)
    * [Seaborn](https://seaborn.pydata.org/)
  - [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows) and [Anakonda](https://www.anaconda.com/products/individual) as an IDE  
  

# Project brief:
- Preprocessing steps:
    * Load the Data
    * Discovering the Data
    * Handling Missing Values
    * Dealing with Outliers
    * Multicollinearity - a phenomenon in which two or more predictor variables 
						  in a multiple regression model are highly correlated, 
						  so that the coefficient estimates may change erratically 
						  in response to small changes in the model or data.
    * Dealing with Categorical Values
    * Standardization
- Processing:
    * k-Nearest Neighbors
- Postprocessing metrics:
    * Precision
    * Recall
    * F1 Score
    * ROC AUC



# Load the Data (default dataset):
```sh
raw_data = pd.read_csv('data/car_ad.csv', encoding="ISO-8859-1")
raw_data.head()
```
|    | car           |   price | body      |   mileage |   engV | engType   | registration   |   year | model   | drive   |
|---:|:--------------|--------:|:----------|----------:|-------:|:----------|:---------------|-------:|:--------|:--------|
|  0 | Ford          |   15500 | crossover |        68 |    2.5 | Gas       | yes            |   2010 | Kuga    | full    |
|  1 | Mercedes-Benz |   20500 | sedan     |       173 |    1.8 | Gas       | yes            |   2011 | E-Class | rear    |
|  2 | Mercedes-Benz |   35000 | other     |       135 |    5.5 | Petrol    | yes            |   2008 | CL 550  | rear    |
|  3 | Mercedes-Benz |   17800 | van       |       162 |    1.8 | Diesel    | yes            |   2012 | B 180   | front   |
|  4 | Mercedes-Benz |   33000 | vagon     |        91 |  nan   | Other     | yes            |   2013 | E-Class | nan     |


# Discovering the Data:
```sh
raw_data.describe(include='all')
```
|        | car        |     price | body   |   mileage |       engV | engType   | registration   |       year | model   | drive   |
|:-------|:-----------|----------:|:-------|----------:|-----------:|:----------|:---------------|-----------:|:--------|:--------|
| count  | 9576       |   9309    | 9576   | 9576      | 9142       | 9576      | 9576           | 9576       | 9576    | 9065    |
| unique | 87         |    nan    | 6      |  nan      |  nan       | 4         | 2              |  nan       | 888     | 3       |
| top    | Volkswagen |    nan    | sedan  |  nan      |  nan       | Petrol    | yes            |  nan       | E-Class | front   |
| freq   | 936        |    nan    | 3646   |  nan      |  nan       | 4379      | 9015           |  nan       | 199     | 5188    |
| mean   | nan        |  16081.7  | nan    |  138.862  |    2.64634 | nan       | nan            | 2006.61    | nan     | nan     |
| std    | nan        |  24301.9  | nan    |   98.6298 |    5.9277  | nan       | nan            |    7.06792 | nan     | nan     |
| min    | nan        |    259.35 | nan    |    0      |    0.1     | nan       | nan            | 1953       | nan     | nan     |
| 25%    | nan        |   5400    | nan    |   70      |    1.6     | nan       | nan            | 2004       | nan     | nan     |
| 50%    | nan        |   9500    | nan    |  128      |    2       | nan       | nan            | 2008       | nan     | nan     |
| 75%    | nan        |  17000    | nan    |  194      |    2.5     | nan       | nan            | 2012       | nan     | nan     |
| max    | nan        | 547800    | nan    |  999      |   99.99    | nan       | nan            | 2016       | nan     | nan     |

