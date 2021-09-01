import streamlit as st
import pandas as pd 
import numpy as np 
from st_aggrid import AgGrid
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import joblib 
import pickle
label_encoder = LabelEncoder()


#// TODO display dataset of training : 
data = pd.read_csv('E:/DATA SCIENCE/Project Web App Classification/model_train/dataset/travel.csv')

st.write('Original Dataset')
AgGrid(data)


#// TODO create form for input 
with st.form('my form') : 
    st.write('To predict whether your customer is willing to buy our insurance product please input data below')
    age_form = st.number_input('Please input customer Age',min_value=1)
    employment_form = st.selectbox('Employment Type',options=('Government Sector', 'Private Sector/Self Employed'))
    graduate_form = st.selectbox('Are you graduate ? ',options=('Yes','No'))
    annual_income_form = st.number_input('Please input yout annual income',min_value=1)
    family_numbers_form = st.number_input('How many family members you have ?',min_value=0)
    chronicdiseases_form = st.selectbox('Do you have chronic disease ? ',options=('Yes','No'))
    frequentFlyer_form = st.selectbox('Do you frequently go travelling ? ',options=('Yes','No'))
    evertravelledAbroad_form = st.selectbox('Do you used to go abroad ? ',options=('Yes','No'))
    submit_btn = st.form_submit_button()
    
#// TODO save all the data to create dataframe
#// TODO load model 
#// TODO preprocess input data 
if submit_btn : 
    st.write('Thank you for your input the models are about to tell you')
    response_data = {'Age' :age_form, 'Employment Type':employment_form,
                     'GraduateOrNot':graduate_form, 'AnnualIncome':annual_income_form,
                     'FamilyMembers':family_numbers_form, 'ChronicDiseases':chronicdiseases_form, 
                     'FrequentFlyer':frequentFlyer_form,'EverTravelledAbroad':evertravelledAbroad_form}
    response_df = pd.DataFrame(response_data)
    # start to preprocess 
    
    def preprocess(data): 
        '''
        data-> dataframe
        '''
        #preprocess 'FrequentFlyer','EverTravelledAbroad','ChronicDiseases column 
        for column in ['FrequentFlyer','EverTravelledAbroad','ChronicDiseases'] : 
            # storing the result of labelling in store_tmp
            store_tmp = label_encoder.fit_transform(data[col])
            data[col] = store_tmp
        #preprocess the remaining categorical columns by using pd.get_dummies
        split_data = data[['Employment Type','GraduateOrNot']]
        initial_data = data.drop(['Employment Type','GraduateOrNot'],axis=1)
        #one hot encoding separated data
        split_data = pd.get_dummies(split_data)
        #joining the data again 
        data = pd.concat([initial_data,split_data],axis=1)
        return data
    model_input = preprocess(response_df)
    #loading the trained_model 
    clf_model = pickle.load('E:/DATA SCIENCE/Project Web App Classification/model_train/classifier_model.pkl')
    
        











