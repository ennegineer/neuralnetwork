# Deep Learning - Charity Funding Predictor
 Predicting with neural networks

## Background

The non-profit foundation Alphabet Soup wants to create an algorithm to predict whether or not applicants for funding will be successful. With knowledge of machine learning and neural networks, we can use the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if they are funded by Alphabet Soup.

From Alphabet Soup’s business team, we have a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* **EIN** and **NAME**—Identification columns
* **APPLICATION_TYPE**—Alphabet Soup application type
* **AFFILIATION**—Affiliated sector of industry
* **CLASSIFICATION**—Government organization classification
* **USE_CASE**—Use case for funding
* **ORGANIZATION**—Organization type
* **STATUS**—Active status
* **INCOME_AMT**—Income classification
* **SPECIAL_CONSIDERATIONS**—Special consideration for application
* **ASK_AMT**—Funding amount requested
* **IS_SUCCESSFUL**—Was the money used effectively
## Process

In this project we are using pandas, sklearn, and tensorflow libraries in Python to train neural network models on the data given to make predictions. 

### Preprocess the data
First we need to preprocess the dataset.
1. Read in the charity_data.csv to a Pandas DataFrame, and identify the targets and features in the dataset:
  * Model target: `IS_SUCCESSFUL` since we are trying to predict whether or not applicants for funding will be successful.
  * Model features: `APPLICATION_TYPE`, `AFFILIATION`,`CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, and `INCOME_AMT`.
2. Drop the `EIN` and `NAME` columns - we don't need them.
3. Look at the total unique values per column and bin the less common values for columns with more than 10 unique values.
4. Use `pd.get_dummies()` to encode categorical variables into numeric values.
5. Fit and scale the data using `StandardScaler()`.

### Compile, Train, Evalute
use tensorflow

### Optimize the model


## Analysis

Below is a report on the performance of the deep learning model we created in this exercise.
### Overview
Explain the purpose of this analysis.
### Results
Using bulleted lists and images to support answers, address these questions:

  * Data Preprocessing
    * What variable(s) are considered the target(s) for your model?
    * What variable(s) are considered to be the features for your model?
    * What variable(s) are neither targets nor features, and should be removed from the input data?
  * Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take to try and increase model performance?

### Summary
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.