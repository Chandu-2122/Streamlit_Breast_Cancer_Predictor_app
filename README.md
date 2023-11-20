# Streamlit_Breast_Cancer_Predictor_app
Predicts based on cell nuclei measurements

# Objective
Build a app that predicts cancer by showing the visualization based on the given cell nuclei measurements

# Dataset Description
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.

Attribute Information:

1) ID number
2) Diagnosis (M = malignant, B = benign)
3-32)

Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter

d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three
largest values) of these features were computed for each image,
resulting in 30 features. For instance, field 3 is Mean Radius, field
13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

# Project Structure
It has 3 folders:

1. **app**:
   
   -> **main.py**: contains the main function of the app

2. **data**:
   
   -> **data.csv**: contains the dataset

3. **model**:
   
   -> **main.py**: contains logistic regression model code to predict

   -> **model.pkl**: pickle file of the model

   -> **scaler.pkl**: pickle file of the scaler
 
# Data Preparation
The dataset is cleaned by 

removing the unwanted columns for prediction: 'id', 'Unnamed: 32',

converting the output objects to integers

and scaling the values using StandardScaler


# Libraries Used
1. **pandas**: used to clean the dataset byy removing unnecessary columns and transforming categorical variables into numerical ones
2. **StandardScaler**: used to preprocess numerical data before model training to ensure that different features have the same scale
3. **train_test_split**: used to split datasets into training and testing subsets
4. **LogisticRegression**: used to train a logistic regression model on labeled data where the target variable has two classes, predicting the probability of a sample belonging to a certain class
5. **accuracy_score**: used to compare the predicted labels to the true labels and return the accuracy-ratio of correctly predicted samples to the total number of samples
6. **pickle**: used to save machine learning model and scaler object into binary files to use later or in different environment
7. **streamlit**: used to create the user interface, generate sliders, visualize graphs and display predictions
8. **plotly.graph_objects**: used to create interactive plots and charts
9. **numpy**: used for numerical operations in data preperation

# Visualization
Radar charts are useful for displaying multiple quantitative variables.

So, plotly's radar chart('go.ScatterPolar') is used for visualizing the mean, standard error and worst values of various features related to breast cancer diagnosis.

# App Snippet 
![image](https://github.com/Chandu-2122/Streamlit_app/assets/107211229/cf727799-6289-4688-bbb9-0c340b33b925)


# Results
Undeployed streamlit app is built which visualizes the input values from a slider using a radar chart to represent mean, standard error and worst values for various features related to breast cancer diagnosis and then the predictions are displayed on the right side of the app, along with the probabilities of being benign or malignant based on logistic regression model training on the input.

# Conclusion
Streamlit framework made easy to build web application for machine learning by simplifying the creation of interactive and data-driven apps.




   
       
