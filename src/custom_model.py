import pandas as pd 
from sklearn import model_selection,metrics
import streamlit as st 
import plotly.graph_objects as Go
from plotly.subplots import make_subplots
import numpy as np 

def start_training(model) : 
    data = pd.read_csv('src/data/preprocessed_data.csv').reset_index(drop=True)
    target = 'TravelInsurance'
    X = data.drop(target,axis=1).values
    y = data[target].values.ravel()
    #splitting dataset 
    X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,y,test_size=0.2)
    cv = model_selection.StratifiedKFold(n_splits=5)
    data_index = []
    acc_score = []
    roc_score = []
    for idx,(train_idx,test_idx) in enumerate(cv.split(X_train,Y_train),start=1) : 
        data_index.append(idx)
        X_TRAIN = X_train[train_idx]
        Y_TRAIN = Y_train[train_idx]
        X_VAL = X_train[test_idx]
        Y_VAL = Y_train[test_idx]
        model.fit(X_TRAIN,Y_TRAIN)
        prediction = model.predict(X_VAL)
        pred_proba = model.predict_proba(X_VAL)[:,1]
        
        accuraction_metrics = metrics.accuracy_score(Y_VAL,prediction)
        acc_score.append(accuraction_metrics)
        roc_metrics = metrics.roc_auc_score(Y_VAL,pred_proba)
        roc_score.append(roc_metrics)
        
        st.write(f"loop={idx} HAS {accuraction_metrics} accuracy score and {roc_metrics} and ROC AUC Score ")
    st.success(f"Training Process is done, the model has {np.mean(acc_score)} average accuracy score and {np.mean(roc_score)} average ROC AUC Score")
    result_df = pd.DataFrame({
        "Loop" : data_index, 
        "Accuracy_Score" : acc_score , 
        "Roc_Score" : roc_score
    },index=data_index)
    model.fit(X_train,Y_train)
    return result_df,model

def generate_chart(result_df) : 
    figure = make_subplots(specs=[[{"secondary_y": True}]]) 
    
    #adding scatter plot to figure object 
    figure.add_trace(
        Go.Scatter(x=result_df['Loop'],y=result_df['Accuracy_Score'],name='Accuracy Score'),secondary_y=False
    )
    figure.add_trace(
        Go.Scatter(x=result_df['Loop'],y=result_df['Roc_Score'],name='ROC AUC Score'),secondary_y=True
    )
    figure.update_layout(title="Metrics over Cross Validation")
    figure.update_yaxes(title_text="<b>Accuracy</b>", secondary_y=False)
    figure.update_yaxes(title_text="<b>ROC AUC</b>", secondary_y=True)
    
    return figure 
        
        
        
        