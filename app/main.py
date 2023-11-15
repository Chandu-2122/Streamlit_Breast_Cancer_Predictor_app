import streamlit as st
import pickle 
import pandas as pd
import plotly.graph_objects as go
import numpy as np

#get the clean data
def get_clean_data():
    #importing the dataset
    data = pd.read_csv("data/data.csv")
    
    #removing unwanted columns
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    
    #transforming the output values to integers
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    
    #return the cleaned data
    return data

#creating the sidebar
def add_sidebar():
    
    #setting the title
    st.sidebar.header('Cell Nuclei Measurements')

    #fetch the clean data
    data = get_clean_data()
    
    #setting the labels
    slider_labels = [
        #label, key
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    #creating dictionary with key and value pairs for keys in slider labels
    input_dict = {}
    
    for label, key in slider_labels:
        #setting the slider parameters and storing the selected value from the slider with the respective key in the dictionary
        input_dict[key] = st.sidebar.slider(label, min_value = float(0), max_value = float(data[key].max()), value = float(data[key].mean()))
        
    #return dictionary
    return input_dict

#scaling the values
def get_scaled_data(input_dict):
    
    #fetching the data
    data = get_clean_data()
    
    #selecting input features 
    X = data.drop(['diagnosis'], axis=1)
    
    #scaling the values
    scaled_dict = {}
    
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
        
    #return scaled dictionary
    return scaled_dict

#visualization
def get_visualization(input_data):
    
    #fetch the user's input values and scale them
    input_data = get_scaled_data(input_data)
    
    #features for visualization
    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']
    
    #plotly graph object
    fig = go.Figure()
    
    #plotting the mean values
    fig.add_trace(go.Scatterpolar(
        #plot these,
        r = [
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
            ],
        
        #on these
        theta=categories,
        
        #color fill
        fill='toself',
        
        #legend
        name='Mean Value'
    ))
    
    #plotting the standard error values
    fig.add_trace(go.Scatterpolar(
        #plot these,
        r = [
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
            ],
        
        #on these
        theta=categories,
        
        #color fill
        fill='toself',
        
        #legend
        name='Standard Error'
    ))
    
    #plotting the worst values
    fig.add_trace(go.Scatterpolar(
        #plot these, 
        r = [
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
            ],
        
        #on these
        theta=categories,
        
        #color fill
        fill='toself',
        
        #legend
        name='Worst Value'
    ))
    
    #setting the parameters
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True
    )
  
    #display the figure
    return fig

#predicting the output and displaying them
def add_predictions(input_data):
    
    #fetching the model.pkl file
    model = pickle.load(open("model/model.pkl", "rb"))
    
    #fetching the scaler.pkl file
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    #storing the user's input values
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    #scaling the input values
    input_array_scaled = scaler.transform(input_array)
    
    #predicting the output
    prediction = model.predict(input_array_scaled)
    
    #displaying the result
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is: ")
    
    if prediction[0]==0:
        st.write("Benign")
        
    else:
        st.write("Malicious")
        
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    #setting the page configuration parameters
    st.set_page_config(
        #title of the page
        page_title="Breast Cancer Predictor",
        
        #icon
        page_icon="👩‍⚕️",
        
        #layout
        layout = "wide",
        
        #initial state of sidebar
        initial_sidebar_state="expanded"
    )
    
    #displaying the sidebar
    input_data = add_sidebar()
    
    #setting the container
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")
    
    #dividing the container into 2 columns in 4:1 ratio
    col1, col2 = st.columns([4,1])
  
    #displaying visualization in column 1
    with col1:
        radar_chart = get_visualization(input_data)
        st.plotly_chart(radar_chart)
        
    #displaying predictions in column 2
    with col2:
        add_predictions(input_data)
        
#main program
if __name__ == '__main__':
    main()
        
    
    