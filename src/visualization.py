import streamlit as st 
import matplotlib.pyplot as plt 
import seaborn as sns 






def visualize_categorical_data(data) : 
    diagram1 = sns.countplot(x='TravelInsurance',data=data, palette=['#3c7ade',"#3cdece"])
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

    diagram2= sns.countplot(x='TravelInsurance',hue='ChronicDiseases',data=data, palette=['#3c7ade',"#3cdece"])
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
    diagram3= sns.countplot(x='TravelInsurance',hue='GraduateOrNot',data=data, palette=['#3c7ade',"#3cdece"])
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

    diagram4= sns.countplot(x='TravelInsurance',hue='FrequentFlyer',data=data, palette=['#3c7ade',"#3cdece"])
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

    diagram5= sns.countplot(x='TravelInsurance',hue='EverTravelledAbroad',data=data, palette=['#3c7ade',"#3cdece"])
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
    
def visualize_numerical_data(data) : 
        st.subheader('Exploratory Data Analysis : Numerical Data')
        st.markdown ('Annual Income Distribution',unsafe_allow_html=True)
        sns.violinplot(y='AnnualIncome',x='TravelInsurance',data=data, palette=['#3c7ade',"#3cdece"])
        st.pyplot()
        st.write('<b>Age Distribution</b>',unsafe_allow_html=True)
        sns.boxplot(y='Age',x='TravelInsurance',data=data, palette=['#3c7ade',"#3cdece"])
        st.pyplot()
        sns.scatterplot(y='AnnualIncome',x='Age',hue='TravelInsurance',data=data, palette=['#3c7ade',"#3cdece"])
        st.pyplot()