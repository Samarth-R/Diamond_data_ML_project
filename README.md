# End to End Machine learning project - Diamond Price Prediction - Samarth R Bogar

## Problem statement

In this project,a Diamond dataset is used, which contains the prices and other attributes of almost 54,000 diamonds

This data set contains:

- `price` **(Target Variable)** : price in US dollars ($326-$18,823)

- `carat` : weight of the diamond (0.2--5.01)

- `cut` : quality of the cut (Fair, Good, Very Good, Premium, Ideal)

- `color` : diamond colour, from J (worst) to D (best)

- `clarity` : a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

- `x` : length in mm (0--10.74)

- `y` : width in mm (0--58.9)

- `z` : depth in mm (0--31.8)

- `depth` : total depth percentage = z / mean(x, y) = 2 \* z / (x + y) (43--79)

- `table` : width of top of diamond relative to widest point (43--95)

## Project workflow

1. Data Ingestion :

   - In Data Ingestion stage the data is first read as csv.
   - This data is then split into training and testing sets, and saved as csv files.

2. Data Transformation :

   - In this stage a ColumnTransformer Pipeline is created.
   - For Numeric Variables first SimpleImputer is applied with strategy median , then Standard Scaling is performed.
   - For Categorical Variables SimpleImputer is applied with most frequent strategy, then one hot encoding is performed, then the data is scaled with Standard Scaler with mean as False.
   - This ColumnTransformer is saved as preprocessor in a pickle file.

3. Model Training :

   - In this stage, all the models are trained along with hyperparameter tuning.
   - After this, a best model among them is selected based on r2_score.
   - This model is saved as pickle file.

4. Prediction Pipeline :

   - This pipeline converts the given data into a dataframe and loads the best model pickle file to predict the final results for the inputs.

5. Flask App creation :
   - A Flask app is designed with a user interface to anticipate gemstone prices within a web application.

_The project aslo contains a Jupyter notebook with Exploratory Data Analysis and Model Training in the 'notebook' folder_
