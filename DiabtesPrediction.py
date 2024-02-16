import os
from pathlib import Path
import pickle
import numpy as np
import wikipedia
import pandas as pd
import streamlit as st
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import streamlit.components.v1 as com
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,classification_report,confusion_matrix

st.markdown('''
   <style>
   .css-6kekos.edgvbvh9{
   background-color:red;
   color:white;
   font-family:calibri;
   font-weight:600;
   }
   </style>
   ''', unsafe_allow_html=True)


st.subheader("Medical patient Record")
col1,col2 = st.columns(2)
with col1:
    input1 = st.text_input('Age',key=1)
    input2 = st.text_input('Gender',key=2)
    input3 = st.text_input('Polyuria',key=3)
    input4 = st.text_input('Polydipsia',key=4)
    input5 = st.text_input('sudden weight loss',key=5)
    input6 = st.text_input('weakness',key=6)
    input7 = st.text_input('Polyphagia',key=7)
    input8 = st.text_input('Genital thrush',key=8)
with col2:
    input9 = st.text_input('visual blurring',key=9)
    input10 = st.text_input('Itching',key=10)
    input11 = st.text_input('Irritability',key=11)
    input12 = st.text_input('delayed healing',key=12)
    input13 = st.text_input('partial paresis',key=13)
    input14 = st.text_input('muscle stiffness',key=14)
    input15 = st.text_input('Alopecia',key=15)
    input16 = st.text_input('Obesity',key=16)

if st.button("Predict Medical Status"):
    data1 = [input1]
    data2 = [input2]
    data3 = [input3]
    data4 = [input4]
    data5 = [input5]
    data6 = [input6]
    data7 = [input7]

    data8  =  [input8]
    data9  =  [input9]
    data10 = [input10]
    data11 = [input11]
    data12 = [input12]
    data13 = [input13]
    data14 = [input14]
    data15 = [input15]
    data16 = [input16]

    da1 = pd.Series(data1)
    da2 = pd.Series(data2)
    da3 = pd.Series(data3)
    da4 = pd.Series(data4)
    da5 = pd.Series(data5)
    da6 = pd.Series(data6)
    da7 = pd.Series(data7)

    da8 = pd.Series(data8)
    da9 = pd.Series(data9)
    da10 = pd.Series(data10)
    da11 = pd.Series(data11)
    da12 = pd.Series(data12)
    da13 = pd.Series(data13)
    da14 = pd.Series(data14)
    da15 = pd.Series(data15)
    da16 = pd.Series(data16)

    frame = {"Age":da1,"Gender":da2,"Polyuria":da3,
             "Polydipsia":da4,"sudden weight loss":da5,"weakness":da6,
             "Polyphagia":da7, "Genital thrush": da8, "visual blurring": da9, "Itching": da10,
             "Irritability": da11, "delayed healing": da12, "partial paresis": da13,
             "muscle stiffness":da14,"Alopecia":da15,"Obesity":da16
             }
    dataframe = pd.DataFrame(frame)
    st.write(dataframe)
    file_path1 = Path(__file__).parent / "model2.pkl"
    with file_path1.open("rb") as f1:
        testmodel = pickle.load(f1)

    # labelInit = LabelEncoder()
    # objectList = dataframe.select_dtypes(include="object").columns
    # for feature in objectList:
    #     dataframe[feature] = labelInit.fit_transform(dataframe[feature].astype(str))
    # print(dataframe)
    # result = testmodel.predict(dataframe)
    # predicted_data = np.array(result)
    # dataframe2 = pd.DataFrame(data=predicted_data, columns=['PredictedData'])
    # st.write(dataframe2)
    # dff5 = int(dataframe2['PredictedData'].to_numpy())

    dataframe = dataframe.replace(['Male', 'Female', 'Yes', 'No'], (1, 0, 1, 0))
    print(dataframe)
    result = testmodel.predict(dataframe)
    predicted_data = np.array(result)
    print(predicted_data)
    dataframe2 = pd.DataFrame(data=predicted_data, columns=['PredictedData'])
    dff5 = int(dataframe2['PredictedData'].to_numpy())

    if dff5 == 1:

            st.error(f"The prediction result for your diabetes test is Positive. It means you do have diabetes. The main reason"
                       f" for getting these result is that you have a very bad medical status on the condition that is listed below."
                       f" N.B As a recommendattion, You have to work on these status by tracking the 5 condition listed below, these is because they are the 5 most important features that could determine the result of the prediction.")
            info2 = wikipedia.summary("Symptoms of diabetes", 3)
            sorted_idx = testmodel.best_estimator_.feature_importances_.argsort()[-5:]
            series1 = pd.Series(dataframe.columns[sorted_idx])
            series2 = pd.Series((testmodel.best_estimator_.feature_importances_[sorted_idx]) * 100)
            frame1 = pd.DataFrame({'Top 5 Best sympthoms which determines the result of the prediction': series1,
                                   'Result of the feature importance out of 100%': series2})
            coll5,coll6 = st.columns(2)
            with coll5:
                st.write(frame1.sort_values(by=['Result of the feature importance out of 100%'], axis=0,
                                            ascending=False))
            with coll6:
                fig = plt.figure(figsize=(20, 10))
                sorted_idx = testmodel.best_estimator_.feature_importances_.argsort()[-5:]
                plt.barh(dataframe.columns[sorted_idx], testmodel.best_estimator_.feature_importances_[sorted_idx])
                st.pyplot(fig)
            print(info2)
            
            st.write('---')
            coll3,coll4 = st.columns(2)
            with coll3:
                st.subheader("Disease Description")
                st.write(info2)
            with coll4:
                query1 = "Diabetes"
                wp_page = wikipedia.page(query1)
                list_img_urls = wp_page.images

                tab1, tab2, tab3 = st.tabs(['Related image 1','Related image 2','Related images 3'])
                with tab1:
                    st.image(list_img_urls[7])
                with tab2:
                    st.image(list_img_urls[8])
                with tab3:
                        st.image(list_img_urls[4])


                
    elif dff5 == 0:
            st.success(f"The prediction result for your diabetes test is Negative. It means you do not have diabetes. The main reason"
                       f" for getting these result is that you have a good medical status on the condition that is listed below."
                       f" N.B As a recommendattion, You have to maintain these status by tracking the 5 condition listed below, these is because they are the 5 most important features that could determine the result of the prediction.")
            sorted_idx = testmodel.best_estimator_.feature_importances_.argsort()[-5:]
            series1 = pd.Series(dataframe.columns[sorted_idx])
            series2 = pd.Series((testmodel.best_estimator_.feature_importances_[sorted_idx]) * 100)
            frame1 = pd.DataFrame({'Top 5 Best sympthoms which determines the result of the prediction': series1,
                                   'Result of the feature importance out of 100%': series2})
            coll5,coll6 = st.columns(2)
            with coll5:
                st.write(frame1.sort_values(by=['Result of the feature importance out of 100%'], axis=0,
                                            ascending=False))
            with coll6:
                fig = plt.figure(figsize=(20, 10))
                sorted_idx = testmodel.best_estimator_.feature_importances_.argsort()[-5:]
                plt.barh(dataframe.columns[sorted_idx], testmodel.best_estimator_.feature_importances_[sorted_idx])
                st.pyplot(fig)


    # if dff5 == 1:
    #     st.error(f"Patient has Diabetes, The status shows that {input2} is positive")
    #     info = wikipedia.summary("What is Diabetes?", 1,auto_suggest = True)
    #     info2 = wikipedia.summary("Symptoms of diabetes",3)

    #     print(info)

    #     query1 = "Diabetes"
    #     wp_page = wikipedia.page(query1)
    #     list_img_urls = wp_page.images
    #     st.image(list_img_urls[1])
    #     st.image(list_img_urls[2])
    #     st.image(list_img_urls[3])

    #     st.subheader("Disease Description")
    #     st.write(info)
    #     st.subheader("Disease Prevention")
    #     st.write(info2)
    # elif dff5 == 0:
    #     st.success(f"Pateint has no Diabetes, The status shows that {input2} is Negative")







