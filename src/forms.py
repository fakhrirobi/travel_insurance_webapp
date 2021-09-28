import streamlit as st 
import pandas as pd 
import numpy as np 



class FormFlow : 
    def __init__(self,response_data,encoder) :
        self.response_data = response_data
        self.label_encoder = encoder
        
        
    @st.cache
    def preprocess_input(self): 
        '''
        data-> dataframe
        '''


        response_df = pd.DataFrame(self.response_data,index=[1])
        
        #preprocess 'FrequentFlyer','EverTravelledAbroad','ChronicDiseases column 
        for column in ['FrequentFlyer','EverTravelledAbroad','ChronicDiseases'] : 
            # storing the result of labelling in store_tmp
            store_tmp = self.label_encoder.fit_transform(response_df[column])
            response_df[column] = store_tmp
        #scaling numerical feature by log10
        num_feature = ['Age','AnnualIncome','FamilyMembers','mean_income_per_member']
        for col in num_feature : 
            response_df.loc[:,col] = np.log10(response_df.loc[:,col])
        #filling one hot encoded feature 
        def fill_employment() :
            if self.response_data['Employment Type_Government Sector'] =='Government Sector': 
                response_df['Employment Type_Government Sector'] = 1
                response_df['Employment Type_Private Sector/Self Employed']=0
            else : 
                response_df['Employment Type_Government Sector'] = 0
                response_df['Employment Type_Private Sector/Self Employed']=1
                
            return response_df

        def fill_graduate() :
            if self.response_data['GraduateOrNot_Yes'] =='Yes': 
                response_df['GraduateOrNot_Yes'] = 1
                response_df['GraduateOrNot_No']=0
            else : 
                response_df['GraduateOrNot_Yes'] = 0
                response_df['GraduateOrNot_No']= 1
                
            return response_df
        
        response_df = fill_employment()
        response_df = fill_graduate()
        
        return response_df