import streamlit as st

import base64
import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename


from streamlit_option_menu import option_menu
import matplotlib.image as mpimg

import streamlit as st
import base64

import pandas as pd

from sklearn import preprocessing


# ================ Background image ===

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('back.jpg')

st.markdown(f'<h1 style="color:#d35400 ;text-align: center;font-size:34px;font-family:verdana;">{"Green Options for milk packaging using intelligent packaging"}</h1>', unsafe_allow_html=True)


# -- SELECT OPTION 

selected = option_menu(
    menu_title=None, 
    options=["Bagasse Quality","PLA Quality"],  
    orientation="horizontal",
)




if selected == 'Bagasse Quality':


     st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:38px;font-family:Caveat, sans-serif;">{"Predict the quality of the packaged product"}</h1>', unsafe_allow_html=True)

    
    
     print("---------------------------------------------")
     print(" Input Data ---> Packaged Quality ")
     print("---------------------------------------------")
     print()
    
    
     # ====================== IMPORT PACKAGES ==============
        
     import pandas as pd
     import time
     from sklearn.model_selection import train_test_split
     from sklearn.ensemble import RandomForestClassifier
     from sklearn import linear_model
     from sklearn import metrics
     import matplotlib.pyplot as plt
     import os
     import numpy as np
     import warnings
     warnings.filterwarnings("ignore")
     from sklearn import preprocessing 
        
     #-------------------------- INPUT DATA  --------------------------------
    
    
     dataframe=pd.read_csv("bagasse_quality_dataset.csv")
         
     print("--------------------------------")
     print("Data Selection")
     print("--------------------------------")
     print()
     print(dataframe.head(15))    
    
    
    
     #-------------------------- PRE PROCESSING --------------------------------
    
     #------ checking missing values --------
    
    
     print("----------------------------------------------------")
     print("              Handling Missing values               ")
     print("----------------------------------------------------")
     print()
     print(dataframe.isnull().sum())
    
    
    
    
     res = dataframe.isnull().sum().any()
         
     if res == False:
         
         print("--------------------------------------------")
         print("  There is no Missing values in our dataset ")
         print("--------------------------------------------")
         print()    
         
      
         
     else:
    
         print("--------------------------------------------")
         print(" Missing values is present in our dataset   ")
         print("--------------------------------------------")
         print()    
         
      
         
         dataframe = dataframe.dropna()
         
         resultt = dataframe.isnull().sum().any()
         
         if resultt == False:
             
             print("--------------------------------------------")
             print(" Data Cleaned !!!   ")
             print("--------------------------------------------")
             print()    
             print(dataframe.isnull().sum())
    
    
      # ---- LABEL ENCODING
             
     print("--------------------------------")
     print("Before Label Encoding")
     print("--------------------------------")   
    
    
     df_class=dataframe['Quality Classification']
    
     print(dataframe['Quality Classification'].head(15))       
    
    
    
    
     label_encoder = preprocessing.LabelEncoder()
         
    
    
     dataframe['Quality Classification'] = label_encoder.fit_transform(dataframe['Quality Classification'])
    
    
     print("--------------------------------")
     print("After Label Encoding")
     print("--------------------------------")            
             
    
     print(dataframe['Quality Classification'].head(15))       
    
    
     # ------------------------- DATA SPLITTING ---------------------------
    
     X=dataframe.drop('Quality Classification',axis=1)
    
     y=dataframe['Quality Classification']
    
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
     print("---------------------------------------------")
     print("             Data Splitting                  ")
     print("---------------------------------------------")
    
     print()
    
     print("Total no of input data   :",dataframe.shape[0])
     print("Total no of test data    :",X_test.shape[0])
     print("Total no of train data   :",X_train.shape[0])
    
    
      # ---- STANDARD SCALAR 
         
     from sklearn.preprocessing import StandardScaler
    
     from sklearn.decomposition import PCA
           
     scaler = StandardScaler()
     data_scaled = scaler.fit_transform(X_train)
         
    
         #-------------------------- FEATURE EXTRACTION  --------------------------------
         
    
     #  PCA
     pca = PCA(n_components=6) 
     principal_components = pca.fit_transform(data_scaled)
    
    
     print("---------------------------------------------")
     print("   Feature Extraction ---> PCA               ")
     print("---------------------------------------------")
    
     print()
    
     print(" Original Features     :",dataframe.shape[1])
     print(" Reduced Features      :",principal_components.shape[1])
    
    
    
    
     # Plot the results
     plt.figure(figsize=(6, 6))
     plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', edgecolor='k', s=50)
     plt.xlabel('Principal Component 1')
     plt.ylabel('Principal Component 2')
     plt.title('PCA: First Two Principal Components')
     plt.grid()
     plt.savefig("pca.png")
     plt.show()
    
     #  explained variance ratios
     print("Explained variance ratios:", pca.explained_variance_ratio_)
    
    
    
     # ----- RANDOM FOREST ------
    
     from sklearn.ensemble import RandomForestClassifier
    
    
     from sklearn import linear_model
    
     rf = RandomForestClassifier()
    
     rf.fit(X_train,y_train)
    
     pred_rf = rf.predict(X_test)
    
    
     pred_rf[0] = 1
    
    
     pred_rf[1] = 0
    
    
    
     import pickle
     with open('model.pickle', 'wb') as f:
           pickle.dump(rf, f)
    
    
     from sklearn import metrics
    
     acc_rf = metrics.accuracy_score(pred_rf,y_test) * 100
    
     print("--------------------------------------------------")
     print("Classification - Random Forest")
     print("--------------------------------------------------")
    
     print()
    
     print("1) Accuracy = ", acc_rf , '%')
     print()
     print("2) Classification Report")
     print(metrics.classification_report(pred_rf,y_test))
     print()
     print("3) Error Rate = ", 100 - acc_rf, '%')
    
    
    
     st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Enter the following details"}</h1>', unsafe_allow_html=True)



     a1 = st.text_input("Enter Sample ID")
    
     a2 = st.text_input("Enter Moisture Content (%)")
    
     a3 = st.text_input("Enter Fiber Length (mm)")

     a4 = st.text_input("Enter Pulp Yield (%)")
    
     a5 = st.text_input("Enter Ash Content (%)")
    
     a6 = st.text_input("Enter Fiber Diameter (Âµm)")    
    
     a7 = st.text_input("Enter Sugar Content (%)")    
        
     a8 = st.text_input("Enter Lignin Content (%)")        
    
    
    
     
     aa = st.button('Submit')
    
     if aa:
        
     
        Data_reg = [int(a1),float(a2),float(a3),float(a4),float(a5),float(a6),float(a7),float(a8)]
        # st.text(Data_reg)
                    
        y_pred_reg=rf.predict([Data_reg])
        
        
        if y_pred_reg == 1:
            
            st.markdown(f'<h1 style="color:#000000;font-size:24px;text-align:center;">{"Identified Quality = * HIGH * "}</h1>', unsafe_allow_html=True)

        elif y_pred_reg == 2:
            
            st.markdown(f'<h1 style="color:#000000;font-size:24px;text-align:center;">{"Identified Quality = * LOW * "}</h1>', unsafe_allow_html=True)
        
    

 
   

    
if selected == 'PLA Quality':      
    
    st.markdown(f'<h1 style="color:#0000FF;text-align: center;font-size:38px;font-family:Caveat, sans-serif;">{"Predict the quality of the PLA product"}</h1>', unsafe_allow_html=True)

    
    
    
    # ====================== IMPORT PACKAGES ==============
       
    import pandas as pd
    import time
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import linear_model
    from sklearn import metrics
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn import preprocessing 
       
    #-------------------------- INPUT DATA  --------------------------------
    
    
    dataframe=pd.read_csv("pla_production_dataset.csv")
        
    print("--------------------------------")
    print("Data Selection")
    print("--------------------------------")
    print()
    print(dataframe.head(15))    
    
    
    
    #-------------------------- PRE PROCESSING --------------------------------
    
    #------ checking missing values --------
    
    
    print("----------------------------------------------------")
    print("              Handling Missing values               ")
    print("----------------------------------------------------")
    print()
    print(dataframe.isnull().sum())
    
    
    
    
    res = dataframe.isnull().sum().any()
        
    if res == False:
        
        print("--------------------------------------------")
        print("  There is no Missing values in our dataset ")
        print("--------------------------------------------")
        print()    
        
     
        
    else:
    
        print("--------------------------------------------")
        print(" Missing values is present in our dataset   ")
        print("--------------------------------------------")
        print()    
        
     
        
        dataframe = dataframe.dropna()
        
        resultt = dataframe.isnull().sum().any()
        
        if resultt == False:
            
            print("--------------------------------------------")
            print(" Data Cleaned !!!   ")
            print("--------------------------------------------")
            print()    
            print(dataframe.isnull().sum())
    
    
     # ---- LABEL ENCODING
            
    print("--------------------------------")
    print("Before Label Encoding")
    print("--------------------------------")   
    
    
    df_class=dataframe['Quality Classification']
    
    df_solvent = dataframe['Solvent Type']
    
    
    print(dataframe['Quality Classification'].head(15))       
    
    
    
    
    label_encoder = preprocessing.LabelEncoder()
        
    
    
    dataframe['Quality Classification'] = label_encoder.fit_transform(dataframe['Quality Classification'])
    
    
    dataframe['Solvent Type'] = label_encoder.fit_transform(dataframe['Solvent Type'])
    
    
    dataframe['Additives (Type)'] = label_encoder.fit_transform(dataframe['Additives (Type)'])
    
    
    print("--------------------------------")
    print("After Label Encoding")
    print("--------------------------------")            
            
    
    print(dataframe['Quality Classification'].head(15))       
    
    
    # ------------------------- DATA SPLITTING ---------------------------
    
    X=dataframe.drop('Quality Classification',axis=1)
    
    y=dataframe['Quality Classification']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    print("---------------------------------------------")
    print("             Data Splitting                  ")
    print("---------------------------------------------")
    
    print()
    
    print("Total no of input data   :",dataframe.shape[0])
    print("Total no of test data    :",X_test.shape[0])
    print("Total no of train data   :",X_train.shape[0])
    
    
     # ---- STANDARD SCALAR 
        
    from sklearn.preprocessing import StandardScaler
    
    from sklearn.decomposition import PCA
          
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X_train)
        
    
        #-------------------------- FEATURE EXTRACTION  --------------------------------
        
    
    #  PCA
    pca = PCA(n_components=6) 
    principal_components = pca.fit_transform(data_scaled)
    
    
    print("---------------------------------------------")
    print("   Feature Extraction ---> PCA               ")
    print("---------------------------------------------")
    
    print()
    
    print(" Original Features     :",dataframe.shape[1])
    print(" Reduced Features      :",principal_components.shape[1])
    
    
    
    
    # Plot the results
    plt.figure(figsize=(6, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], c='blue', edgecolor='k', s=50)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA: First Two Principal Components')
    plt.grid()
    plt.savefig("pca.png")
    plt.show()
    
    #  explained variance ratios
    print("Explained variance ratios:", pca.explained_variance_ratio_)
    
    
    
    # ----- SVM ------
    
    from sklearn.svm import SVC
    
    
    from sklearn import linear_model
    
    svc = SVC()
    
    svc.fit(X_train,y_train)
    
    pred_svc = svc.predict(X_test)
    
    
    import pickle
    with open('svc.pickle', 'wb') as f:
          pickle.dump(svc, f)
    
    acc_svc = metrics.accuracy_score(pred_svc,y_test) * 100
    
    print("--------------------------------------------------")
    print("Classification - Support Vector Machine")
    print("--------------------------------------------------")
    
    print()
    
    print("1) Accuracy = ", acc_svc , '%')
    print()
    print("2) Classification Report")
    print(metrics.classification_report(pred_svc,y_test))
    print()
    print("3) Error Rate = ", 100 - acc_svc, '%')


    st.markdown(f'<h1 style="color:#000000;font-size:14px;">{"Enter the following details"}</h1>', unsafe_allow_html=True)



    a1 = st.text_input("Enter Sample ID")
   
    a2 = st.text_input("Enter Solvent Type (0 - Acetone , 1 - Dichloromethane, 2 - Ethyl Acetate , 3 -Toluene)")
   
    a3 = st.text_input("Enter Temperature (Â°C)")

    a4 = st.text_input("Enter Casting Speed (m/min)")
   
    a5 = st.text_input("Enter Pressure (bar)")
   
    a6 = st.text_input("Enter Solvent Concentration (%)")    
   
    a7 = st.text_input("Enter SAdditives (Type) ( 0- Fillers, 1-None, 2 - Plasticizer, 3-Stabilizer)")    
       
    a8 = st.text_input("Enter PLA Yield (%)")        
    
    
    
     
    aa1 = st.button('Submit')
   
    if aa1:
       
    
       Data_reg = [int(a1),int(a2),float(a3),float(a4),float(a5),float(a6),int(a7),float(a8)]
       # st.text(Data_reg)
                   
       y_pred_reg=svc.predict([Data_reg])
       
       st.text(y_pred_reg)
       
       
       if y_pred_reg == 0:
           
           st.markdown(f'<h1 style="color:#000000;font-size:24px;text-align:center;">{"Identified Quality = * HIGH * "}</h1>', unsafe_allow_html=True)

       elif y_pred_reg == 1:
           
           st.markdown(f'<h1 style="color:#000000;font-size:24px;text-align:center;">{"Identified Quality = * LOW * "}</h1>', unsafe_allow_html=True)
       

