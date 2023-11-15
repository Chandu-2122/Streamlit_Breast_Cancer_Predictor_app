#installing necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

#data cleaning
def get_clean_data():
    
    #importing the dataset
    data = pd.read_csv("streamlit_cancer_app/data/data.csv")
    
    #removing unwanted columns
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    
    #transforming the output values to integers
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    
    #return the cleaned data
    return data

#creating the model
def create_model(data):
    
    #taking input features
    X = data.drop(['diagnosis'], axis=1)
    
    #taking the output feature
    Y = data['diagnosis']
    
    #scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #splitting the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 22)
    
    #training the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    #testing the model
    y_pred = model.predict(X_test)
    
    #accuracy of the model
    print('Accuracy of the model: ', accuracy_score(Y_test, y_pred))

    #return the scaler and the model
    return model, scaler

#actual main function
def main():
    
    #get the cleaned data
    data = get_clean_data()
    
    #build the model
    model, scaler = create_model(data)
    
    #pickle file of the model
    with open('streamlit_cancer_app/model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    #pickle file of the scaler
    with open('streamlit_cancer_app/model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        

if __name__ == '__main__':
    main()