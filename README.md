# Introduction to Intelligent systems
# Final project
> In this project, we will perform preliminary processing of the data set and then the selected data processing, and finally post-processing the results. 
The project will contain all the necessary comments and explanations. The data will also be visible for better understanding.
In order for it to run non-stop for presentation purposes only, we will assume that some manual action was taken where expected (i.e., data processing and decision making)

# Submitters:
[Andrey Dodon](https://www.kaggle.com/andreydodon) - `Afeka M.Sc. student, Intelligent systems`

[Michael Gudovsky](https://il.linkedin.com/in/michael-gudovsky-1392157b) - `Afeka M.Sc. student, Intelligent systems`


# Resources:
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
```
raw_data = pd.read_csv('data/data.csv')
raw_data.head()
```
|    |   userid |   gender |   age |   salary |   seniority (years) in comapny |   seniority in role(years) |   monthly return on loan |   how many children |   weight |   height |   grade in last year review (0-10) |   averaged grade of the BSC |
|---:|---------:|---------:|------:|---------:|-------------------------------:|---------------------------:|-------------------------:|--------------------:|---------:|---------:|-----------------------------------:|----------------------------:|
|  0 |        1 |        1 |    49 |    14389 |                             14 |                         33 |                     1313 |                   1 |       83 |      196 |                                  6 |                          60 |
|  1 |        2 |        0 |    41 |     9322 |                              9 |                          9 |                      307 |                   3 |       79 |      176 |                                  8 |                          68 |
|  2 |        3 |        1 |    47 |    10354 |                             15 |                          9 |                       14 |                   4 |       73 |      167 |                                 10 |                          71 |
|  3 |        4 |        1 |    47 |    13383 |                             16 |                         16 |                     3009 |                   4 |       86 |      177 |                                  8 |                          72 |
|  4 |        5 |        0 |    36 |     6751 |                              6 |                         13 |                      224 |                   5 |       64 |      162 |                                  8 |                          92 |


# Discovering the Data:
Let's detect missing values if any
```
raw_data.isna().sum()
```
|                                  |count|
|:---------------------------------|----:|
| userid                           |   0 |
| gender                           |   0 |
| age                              |   0 |
| salary                           |   0 |
| seniority (years) in comapny     |   0 |
| seniority in role(years)         |   0 |
| monthly return on loan           |   0 |
| how many children                |   0 |
| weight                           |   0 |
| height                           |   0 |
| grade in last year review (0-10) |   0 |
| averaged grade of the BSC        |   0 |

Let's check the number of non-NA/null observations in the data set
```
raw_data.count()
[8 rows x 12 columns]
```
|                                  |   0 |
|:---------------------------------|----:|
| userid                           | 200 |
| gender                           | 200 |
| age                              | 200 |
| salary                           | 200 |
| seniority (years) in comapny     | 200 |
| seniority in role(years)         | 200 |
| monthly return on loan           | 200 |
| how many children                | 200 |
| weight                           | 200 |
| height                           | 200 |
| grade in last year review (0-10) | 200 |
| averaged grade of the BSC        | 200 |

Let's check the dtypes of the data set
```
raw_data.dtypes
```
|                                  | 0     |
|:---------------------------------|:------|
| userid                           | int64 |
| gender                           | int64 |
| age                              | int64 |
| salary                           | int64 |
| seniority (years) in comapny     | int64 |
| seniority in role(years)         | int64 |
| monthly return on loan           | int64 |
| how many children                | int64 |
| weight                           | int64 |
| height                           | int64 |
| grade in last year review (0-10) | int64 |
| averaged grade of the BSC        | int64 |

All the data seem to be perfectly aligned. Now, let’s discover the data. We can use the describe method – 
if we use this method we will get only the descriptive statistics of the numerical features.
Since all the data in our data set is int64 (numerical) it will perfectly work
```
raw_data.describe(include='all')
```
|       |   userid |     gender |      age |   salary |   seniority (years) in comapny |   seniority in role(years) |   monthly return on loan |   how many children |   weight |   height |   grade in last year review (0-10) |   averaged grade of the BSC |
|:------|---------:|-----------:|---------:|---------:|-------------------------------:|---------------------------:|-------------------------:|--------------------:|---------:|---------:|-----------------------------------:|----------------------------:|
| count | 200      | 200        | 200      |   200    |                      200       |                   200      |                   200    |           200       | 200      | 200      |                          200       |                    200      |
| mean  | 100.5    |   0.475    |  46.27   | 12261.3  |                       13.805   |                    16.915  |                  1499.02 |             2.375   |  70.49   | 168.965  |                            7.44    |                     76.895  |
| std   |  57.8792 |   0.500628 |  13.7683 |  5220.62 |                        7.33183 |                    14.8184 |                  2153.59 |             1.52842 |  14.7389 |  14.0048 |                            3.13136 |                     15.7848 |
| min   |   1      |   0        |  20      |  4700    |                        1       |                     1      |                     0    |             0       |  41      | 132      |                            0       |                     60      |
| 25%   |  50.75   |   0        |  35      |  7206.5  |                        8       |                     5      |                   199.5  |             1       |  60      | 158      |                            5.75    |                     60      |
| 50%   | 100.5    |   0        |  47      | 11707    |                       14       |                    13      |                   585    |             2       |  70      | 167      |                            9       |                     76      |
| 75%   | 150.25   |   1        |  58      | 16608.2  |                       19       |                    25.25   |                  2026.25 |             3       |  79      | 180      |                           10       |                     92.25   |
| max   | 200      |   1        |  70      | 22624    |                       30       |                    70      |                 15096    |             8       | 111      | 204      |                           10       |                    100      |


