from os import write
from src.visualization import visualize_numerical_data,visualize_categorical_data
from src.button import download_button
from src.forms import FormFlow
from src.custom_model import start_training,generate_chart
from st_on_hover_tabs import on_hover_tabs # using on hover tabs custom components streamlit 
import streamlit as st
import plotly.express as px  
import pandas as pd 
import numpy as np 
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import urllib
import xgboost 
import joblib


from st_aggrid import AgGrid

sns.set_style('whitegrid')
st.set_option('deprecation.showPyplotGlobalUse', False)




data = pd.read_csv('src/data/travel.csv')
# tabs = st.sidebar.radio('Page Selector',('Project Explanation','Exploratory Data Analysis','Create your own model','Try Prediction'))
# with st.sidebar:
#     tabs = on_hover_tabs(tabName=['Project Explanation', 'Exploratory Data Analysis', 'Create your own model','Try Prediction'], 
#                          iconName=['Project Explanation', 'Exploratory Data Analysis', 'Create your own model','Try Prediction'], default_choice=0
# )
tab1, tab2, tab3,tab4 = st.tabs(['Project Explanation', 'Exploratory Data Analysis', 'Create your own model','Try Prediction'])

with tab1 :
    st.image('assets/insurance_scott_graham.jpg')
    html_embed = '[Photo by Scott Graham on Unsplash](https://unsplash.com/@homajob?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)'
    st.markdown(html_embed,unsafe_allow_html=True)
  
    url = 'https://raw.githubusercontent.com/fakhrirobi/travel_insurance_webapp/main/README.md'
    def get_file_content_as_string(url):
        #for reading readme.md from github
        response = urllib.request.urlopen(url)
        return response.read().decode("utf-8")
    st.markdown(get_file_content_as_string(url),unsafe_allow_html=True)
    
with tab2 :
    st.title('Exploratory Data Analysis')
    dataset_desc_exp = st.expander('Dataset Description',expanded=True)
    dataset_desc_exp.markdown('''
    |Column Name  |Description  |
    |---------|---------|
    |Age     |   Age Of The Customer      |
    |Employment     | The Sector In Which Customer Is Employed        |
    |GraduateOrNot     |  Whether The Customer Is College Graduate Or Not       |
    |AnnualIncome     |   The Yearly Income Of The Customer In Indian Rupees[Rounded To Nearest 50 Thousand Rupees      |
    |FamilyMembers     |     Number Of Members In Customer's Family    |
    |ChronicDisease     |Whether The Customer Suffers From Any Major Disease Or Conditions Like Diabetes/High BP or Asthama,etc         |
    |FrequentFlyer     |  Derived Data Based On Customer's History Of Booking Air Tickets On Atleast 4 Different Instances In The Last 2 Years[2017-2019]       |
    |EverTravelledAbroad     | Has The Customer Ever Travelled To A Foreign Country[Not Necessarily Using The Company's Services]        |
    |TravelInsurance     |   Did The Customer Buy Travel Insurance Package During Introductory Offering Held In The Year 2019      |
                    
                ''',unsafe_allow_html=True)
    data_expander = st.expander('Data Inspection')
    col1,col2 = data_expander.columns(2)
    with col1 : 
        st.subheader('Missing Values Inspection: ')
        missing = pd.DataFrame({"columns":[x for x in data.columns],
                                "% missing data" :[(x/data.shape[0])*100 for x in data.isnull().sum()] })
        AgGrid(missing)
    with col2 : 
        st.subheader('Dataset Statistics: ')
        AgGrid(data.describe())
    col3,col4 = data_expander.columns(2)
    with col3 : 
        st.subheader('Exploratory Data Analysis : Categorical Data')
        visualize_categorical_data(data)
    with col4 : 
        
        visualize_numerical_data(data)
            
with tab3 : 
    st.subheader("Custom Params for creating XGBoost Model in Travel Insurance dataset")
    with st.form("model_customization") : 
        st.write('please input this following value to customize model')
        xgb_param  = {
            'objective':'binary:logistic',
            'max_depth': 6,
            'alpha': 10,
            'learning_rate': 0.0001,
            'n_estimators':300
        }
        max_depth = st.slider("please input num of max_depth",min_value=1,max_value=50,step=1)
        alpha = st.slider("please input num of alpha",min_value=1,max_value=30,step=1)
        learning_rate = st.slider("please input num of learning_rate",min_value=0.0001,max_value=1.0,step=0.0001)
        n_estimators = st.slider("please input num of n_estimators",min_value=10,max_value=1000,step=10)
        model_name_to_save = st.text_input("Please input the name of your model to")
        custom_params = {
            'objective':'binary:logistic',
            'max_depth': max_depth,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'n_estimators':n_estimators
        }
        start_train_button = st.form_submit_button("Start Training Model")
        
    custom_model = xgboost.XGBClassifier(**custom_params)
    if start_train_button : 
        result,trained_model = start_training(custom_model)
        st.write("Model Result : ")
        AgGrid(result)
        figure = generate_chart(result)
        st.plotly_chart(figure)
        filename = f'{model_name_to_save}.pkl'
        with open(filename,'wb') as file : 
            model_download = joblib.dump(result,file)
        download_btn = download_button(model_download,download_filename=filename,button_text=f'Click here to download {filename}', pickle_it=False)
        st.markdown(download_btn, unsafe_allow_html=True)
        




        
with tab4 : 
    st.title("Let's Try to Predict whether your customer want to buy Travel Insurance")
    with st.form('my form') : 
        st.write('To predict whether your customer is willing to buy our insurance product please input your customer candidate data below')
        age_form = st.number_input('Age',min_value=1)
        employment_form = st.selectbox('Employment Type',options=('Government Sector', 'Private Sector/Self Employed'))
        graduate_form = st.selectbox('are your customer a  graduate ? ',options=('Yes','No'))
        annual_income_form = st.number_input('Please input customer annual income',min_value=1)
        family_numbers_form = st.number_input('How many family members they have ?',min_value=1)
        chronicdiseases_form = st.selectbox('have chronic disease ? ',options=('Yes','No'))
        frequentFlyer_form = st.selectbox('Do they  frequently go travelling ? ',options=('Yes','No'))
        evertravelledAbroad_form = st.selectbox('did they use to go abroad ? ',options=('Yes','No'))
        submit_btn = st.form_submit_button()
    if submit_btn : 
        st.write('Thank you for your input the models are about to tell you')
        response_data={'Age':age_form, 'AnnualIncome':annual_income_form, 
                    'FamilyMembers':family_numbers_form, 'ChronicDiseases':chronicdiseases_form,
                    'FrequentFlyer':frequentFlyer_form, 'EverTravelledAbroad':evertravelledAbroad_form,
                    'mean_income_per_member' : annual_income_form/(family_numbers_form+1),
                    'Employment Type_Government Sector':employment_form,
                    'Employment Type_Private Sector/Self Employed':employment_form, 
                    'GraduateOrNot_No':graduate_form,'GraduateOrNot_Yes':graduate_form}

        from sklearn import preprocessing
        label_encoder = preprocessing.LabelEncoder()
        # start to preprocess 
        form_process = FormFlow(response_data, label_encoder)
        model_input = form_process.preprocess_input()
        
        #loading the trained_model 
        # path = os.path.join(current_dir,')
        filename = r'src/xgb_model_without_tuning.pkl'
        clf_model = joblib.load(filename)
            
        pred = clf_model.predict(model_input)
        proba=clf_model.predict_proba(model_input)
        proba_result = pd.DataFrame(proba,columns=['Will not Buy Travel Insurance','Will Buy Travel Insurance'])
        proba_result = proba_result.apply(lambda x : round(100*x,2))
        if pred == 1 :
            st.title('Prediction Result : ')
            st.success('Your Customer is going to buy Travel Insurance')
        else  : 
            st.title('Prediction Result :') 
            st.warning('Your Customer is  not going to buy Travel Insurance') 

    
        #charting 
        fig = px.bar(x=proba_result.columns,y=proba_result.iloc[0],color=proba_result.columns)
        st.plotly_chart(fig)
        
        
        
     

else : 
    st.write('No Choice Bro')

