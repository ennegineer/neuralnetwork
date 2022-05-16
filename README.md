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
Using tensorflow, we compiled the sequential model with relu activation method and evaluated at 50 epochs. The loss was very high - 55%, and accuracy was below 74%, so we added another hidden layer and tried again. The loss remained about the same. 

### Optimize the model
Next we set up a function to help us find the best options for a sequential model. The function was very taxing on CPU and ran for a long time without finishing, but in the time that it did run, it looked like the tanh activation method was more successful than relu. We re-ran our original sequential model with tanh instead of relu as the activation method. Accuracy and loss remained about the same.

## Analysis

Below is a report on the performance of the deep learning model we created in this exercise.
### Overview
The purpose of this analysis was to attempt to predict whether a loan applicant would be successful if funded, based on the existing data provided by the foundation. We followed standard practices for preprocessing the data, dropping unnecessary columns and fitting and transforming the data after changing categorical values to numeric. 

Next we ran several sequential models to try to find the best accuracy at minimum loss. All of our models produced the same results: approximately 73% accuracy, but at 55% loss.
### Results

  * Data Preprocessing
    * `IS_SUCCESSFUL` is the target for our model, since that is the variable we want to predict.
    * `EIN` and `NAME` are neither targets nor features, and were removed from the input data.
    * All other variables are considered to be the features for the model.
  * Compiling, Training, and Evaluating the Model
    * Our neural network model has 16 neurons with 3 layers, with relu as the activation on two and tanh on the third. Our initial run was with two layers and 16 neurons each, using tanh as the activation method. This did not perform as well as the second model.
    * We were not able to achieve high performance with our models - accuracy in our first few models was around 73%, and with 55% loss. Later models performed much worse, with an accuracy around 46% and loss of 70%.
    * We created a function to attempt several combinations of neurons, layers and activation methods to try to find the best fit. Running this function as keras tuner was very expensive computationally and could not complete on our machine, but we were able to view the results on over 70 trials. None performed better than our first two.

### Summary

Overall, our deep learning model is not ideal. Perhaps a better outcome could be reached if we had more data, looked closely for outliers, and if feasible, run a random forest classification model. I would recommend using random forest because it takes many decision tree models and looks for the majority result.