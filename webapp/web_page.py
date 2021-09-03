import streamlit as st
import pandas as pd 
import numpy as np 
from st_aggrid import AgGrid
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import joblib 
import pickle
import os
import plotly.express as px 
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
label_encoder = LabelEncoder()



st.title('Travel Insurance Buy Prediction App')
#// TODO display dataset of training : 



first_scratch = st.expander('Exploratory Data Analysis')
data = pd.read_csv('E:/DATA SCIENCE/Project Web App Classification/model_train/dataset/travel.csv')
######################################## PREPROCESS DATA
#// TODO : PART INSPECTING MISSING VALUES
col1,col2 = first_scratch.columns(2)
with col1 : 
    st.write('Missing Values Inspection: ')
    st.write(data.isnull().sum())
with col2 : 
    st.write('Data Description: ')
    st.write(data.describe())

col3,col4=first_scratch.columns(2)

with col3 : 
    
    st.write('Exploratory Data Analysis : Categorical Data')
    diagram1 = sns.countplot(x='TravelInsurance',data=data)
    
    st.write('Travel Insurance')
    for patch in diagram1.patches :
        plt.annotate(text=patch.get_height(),
                    xy=(patch.get_x() + patch.get_width() / 2, 
                        patch.get_height()),
                    verticalalignment='center',    # Center the text 
                    horizontalalignment='center',  # from both directions
                    xytext=(0, 15),
                    textcoords='offset points')
    st.pyplot()

    diagram2= sns.countplot(x='TravelInsurance',hue='ChronicDiseases',data=data)
    st.write('Travel Insurance divided by Chronic Disease')    
    for patch in diagram2.patches :
        plt.annotate(text=patch.get_height(),
                    xy=(patch.get_x() + patch.get_width() / 2, 
                        patch.get_height()),
                    verticalalignment='center',    # Center the text 
                    horizontalalignment='center',  # from both directions
                    xytext=(0, 15),
                    textcoords='offset points')
    st.pyplot()
    diagram3= sns.countplot(x='TravelInsurance',hue='GraduateOrNot',data=data)
    st.write('Travel Insurance divided by GraduateOrNot')    
    for patch in diagram3.patches :
        plt.annotate(text=patch.get_height(),
                    xy=(patch.get_x() + patch.get_width() / 2, 
                        patch.get_height()),
                    verticalalignment='center',    # Center the text 
                    horizontalalignment='center',  # from both directions
                    xytext=(0, 15),
                    textcoords='offset points')
    st.pyplot()

    diagram4= sns.countplot(x='TravelInsurance',hue='FrequentFlyer',data=data)
    st.write('Travel Insurance divided by FrequentFlyer')   
    for patch in diagram4.patches :
        plt.annotate(text=patch.get_height(),
                    xy=(patch.get_x() + patch.get_width() / 2, 
                        patch.get_height()),
                    verticalalignment='center',    # Center the text 
                    horizontalalignment='center',  # from both directions
                    xytext=(0, 15),
                    textcoords='offset points')
    st.pyplot()

    diagram5= sns.countplot(x='TravelInsurance',hue='EverTravelledAbroad',data=data)
    st.write('Travel Insurance divided by EverTravelledAbroad')   
    for patch in diagram5.patches :
        plt.annotate(text=patch.get_height(),
                    xy=(patch.get_x() + patch.get_width() / 2, 
                        patch.get_height()),
                    verticalalignment='center',    # Center the text 
                    horizontalalignment='center',  # from both directions
                    xytext=(0, 15),
                    textcoords='offset points')
    st.pyplot()
    
with col4 : 
    st.write('Exploratory Data Analysis : Numerical Data')
    st.write ('Annual Income Distribution')
    sns.violinplot(y='AnnualIncome',x='TravelInsurance',data=data)
    st.pyplot()
    st.write('Age Distribution')
    sns.boxplot(y='Age',x='TravelInsurance',data=data)
    st.pyplot()
second_expander = st.expander('Create your Own Model from this Dataset')



    

#//TODO : PART DESCRIBING THE DATA
#// TODO : PART UNIQUE COLUMN 
# // TODO : EDA 
    #// CATEGORICAL 
    #// NUMERICAL 
#//TODO : DOWNLOAD CLEANED DATA 

######################################## TRAINING PROCESS 
#// TODO : 






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

    
    # start to preprocess 
    @st.cache
    def preprocess_input(): 
        '''
        data-> dataframe
        '''
        response_data={'Age':age_form, 'AnnualIncome':annual_income_form, 
                       'FamilyMembers':family_numbers_form, 'ChronicDiseases':chronicdiseases_form, 
                       'FrequentFlyer':frequentFlyer_form, 'EverTravelledAbroad':evertravelledAbroad_form,
                       'Employment Type_Government Sector':employment_form,
                       'Employment Type_Private Sector/Self Employed':employment_form, 
                       'GraduateOrNot_No':graduate_form,'GraduateOrNot_Yes':graduate_form}
        
        response_df = pd.DataFrame(response_data,index=[1])
        
        #preprocess 'FrequentFlyer','EverTravelledAbroad','ChronicDiseases column 
        for column in ['FrequentFlyer','EverTravelledAbroad','ChronicDiseases'] : 
            # storing the result of labelling in store_tmp
            store_tmp = label_encoder.fit_transform(response_df[column])
            response_df[column] = store_tmp
            
        #filling one hot encoded feature 
        def fill_employment() :
            if employment_form =='Government Sector': 
                response_df['Employment Type_Government Sector'] = 1
                response_df['Employment Type_Private Sector/Self Employed']=0
            else : 
                response_df['Employment Type_Government Sector'] = 0
                response_df['Employment Type_Private Sector/Self Employed']=1
                
            return response_df
        
        def fill_graduate() :
            if graduate_form =='Yes': 
                response_df['GraduateOrNot_Yes'] = 1
                response_df['GraduateOrNot_No']=0
            else : 
                response_df['GraduateOrNot_Yes'] = 0
                response_df['GraduateOrNot_No']= 1
                
            return response_df
        response_df = fill_employment()
        response_df = fill_graduate()
        return response_df
    model_input = preprocess_input()
    #
    
    #loading the trained_model 
    # path = os.path.join(current_dir,')
    filename = r'E:/DATA SCIENCE/travel_insurance_webapp/model_train/classifier_model.pkl'
    with open(filename,'rb') as file :
        clf_model = pickle.load(file)
        
    pred = clf_model.predict(model_input)
    proba=clf_model.predict_proba(model_input)

    
    proba_result = pd.DataFrame(proba,columns=['Will not Buy Travel Insurance','Will Buy Travel Insurance'])
    proba_result = proba_result.apply(lambda x : round(100*x,2))


    
    #charting 
    fig = px.bar(x=proba_result.columns,y=proba_result.iloc[0])
    st.plotly_chart(fig)
    
        











