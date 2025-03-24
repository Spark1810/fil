# in the requirements.txt add scipy


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from scipy.spatial import KDTree

data = {
    'MMSE': [29, 27, 29, 29, 30, 28, 29, 28, 27, 30, 30, 30, 29, 29, 29, 28, 30, 30, 30, 30, 28, 27, 29, 30, 30, 30, 30, 30, 28, 28, 27, 23, 29, 30, 30, 30, 30, 30, 28, 28, 27, 29, 30, 30, 30, 30, 30, 29, 29, 30, 30, 30, 30, 30, 29, 30, 30, 30, 30, 28, 27, 30, 28, 27, 29, 28, 27, 29, 30, 30, 30, 30, 29, 29, 29, 29, 28, 30, 30, 30, 30, 29, 29, 29, 28],
    'eTIV': [1432, 1432, 1548, 1534, 1550, 1511, 1506, 1383, 1390, 1513, 1449, 1769, 1785, 1814, 1154, 1165, 1611, 1628, 1446, 1412, 1475, 1484, 1583, 1586, 1590, 1548, 1619, 1443, 1479, 1507, 1429, 1502, 1491, 1554, 1420, 1452, 1686, 1784, 1569, 1575, 1425, 1688, 1564, 1551, 1550, 1556, 1753, 1777, 1560, 1781, 1472, 1520, 1444, 1517, 1561, 1477, 1579, 1778, 1566, 1441, 1469, 1505, 1522, 1526, 1603, 1481, 1467, 1503, 1568, 1495, 1478, 1576, 1558, 1442, 1490, 1509, 1553, 1549, 1482, 1779, 1562, 1508, 1557, 1473, 1431, 1571],
    'nWBV': [0.692, 0.684, 0.773, 0.772, 0.758, 0.739, 0.715, 0.748, 0.728, 0.771, 0.774, 0.699, 0.687, 0.679, 0.75, 0.736, 0.729, 0.709, 0.78, 0.783, 0.762, 0.75, 0.777, 0.757, 0.76, 0.733, 0.727, 0.748, 0.772, 0.773, 0.764, 0.762, 0.768, 0.752, 0.735, 0.736, 0.685, 0.688, 0.713, 0.711, 0.703, 0.685, 0.731, 0.759, 0.752, 0.74, 0.703, 0.704, 0.741, 0.695, 0.719, 0.728, 0.71, 0.716, 0.737, 0.701, 0.711, 0.706, 0.724, 0.743, 0.745, 0.738, 0.73, 0.74, 0.718, 0.757, 0.762, 0.715, 0.724, 0.739, 0.758, 0.774, 0.75, 0.747, 0.727, 0.755, 0.752, 0.727, 0.712, 0.702, 0.745, 0.739, 0.718, 0.722],
    'ASF': [1.225, 1.226, 1.134, 1.144, 1.133, 1.162, 1.166, 1.269, 1.263, 1.16, 1.212, 0.992, 0.983, 0.967, 1.521, 1.506, 1.089, 1.078, 1.214, 1.243, 1.19, 1.183, 1.108, 1.107, 1.104, 1.131, 1.119, 1.211, 1.134, 1.12, 1.148, 1.156, 1.173, 1.165, 1.212, 1.207, 1.013, 1.015, 1.176, 1.171, 1.186, 1.009, 1.196, 1.156, 1.179, 1.184, 1.005, 1.002, 1.174, 1.01, 1.198, 1.205, 1.199, 1.193, 1.177, 1.01, 1.181, 1.195, 1.181, 1.179, 1.15, 1.149, 1.168, 1.191, 1.18, 1.096, 1.095, 1.168, 1.16, 1.15, 1.171, 1.153, 1.181, 1.142, 1.12, 1.137, 1.147, 1.125, 1.144, 1.145, 1.16, 1.148, 1.145, 1.124, 1.151],
    'Group': ['Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Converted', 'Converted', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Demented', 'Demented', 'Demented', 'Demented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented', 'Nondemented']
}
min_len = min(len(data['MMSE']), len(data['eTIV']), len(data['nWBV']), len(data['ASF']), len(data['Group']))
data = {k: v[:min_len] for k, v in data.items()}
df = pd.DataFrame(data)

features = df[['MMSE', 'eTIV', 'nWBV', 'ASF']].values
kdtree = KDTree(features)

sns.set() 

st.title("Alzheimer Detection ")

activities = ["Introduction", "Statistics", "Prediction", "About Us"]
choice = st.sidebar.selectbox("Select Activities", activities)
if choice == 'Introduction':
    st.image("projectimage.jpg")
    st.markdown(
        "Alzheimer is a term used to describe a group of symptoms affecting memory, thinking and social abilities severely enough to interfere with your daily life. It isn't a specific disease, but several diseases can cause Alzheimer. Though Alzheimer generally involves memory loss, memory loss has different causes")
    st.title("A look into the scientific side of demenetia ")
    st.write("Parameters taken")
    st.write("A major parameters for Alzheimer prediction is MMSE,SES,eTIV,nWBV,ASF")
    st.write("MMSE - Mini Mental State Examination")
    st.write("SES - Social Economic State")
    st.write("eTIV - Estimated Total Intracranial Volume")
    st.write("nWBV - Normalised  Whole Brain Volume")
    st.write("ASF - Atlas Scaling Factor")
    st.write("Each one of those parameters have a particular effect when predicting Alzheimer.")
    
# ==========================================================================================================================

elif choice=="Statistics":
    st.title("Wanna Clarify about your Alzheimer status?")
    
    df = pd.read_csv("oasis_longitudinal.csv")
    
    # Use first visit data only for the analysis
    df = df.loc[df['Visit'] == 1].reset_index(drop=True)
    
    # Replace values in 'M/F' and 'Group' columns
    df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])
    
    # Drop unnecessary columns
    df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)
    
    # Function to create a bar chart
    def bar(feature):
        Demented = df[df['Group'] == 1][feature].value_counts()
        Nondemented = df[df['Group'] == 0][feature].value_counts()
        _bar = pd.DataFrame([Demented, Nondemented])
        _bar.index = ['Demented', 'Nondemented']
        _bar.plot(kind='bar', stacked=True, figsize=(8, 5))
    
    # Gender and Group (Female=0, Male=1)
    bar('M/F')
    plt.xlabel('Group')
    plt.ylabel('Number of patients')
    plt.legend()
    plt.title('Gender v/s Demented rate')
    st.pyplot(plt.gcf())
    
    # Alzheimer Distribution by Gender
    Alzheimer_by_gender = df[df['Group'] == 1]['M/F'].value_counts()
    fig, ax = plt.subplots()
    Alzheimer_by_gender.plot(kind='bar', ax=ax)
    st.subheader('Alzheimer Distribution by Gender :')
    ax.set_title('Alzheimer Distribution by Gender')
    ax.set_xlabel('Gender (Female=0, Male=1)')
    ax.set_ylabel('Number of Patients')
    st.pyplot(fig)
    
    # Dementia Distribution by MMSE
    st.subheader('Dementia Distribution by MMSE :')
    fig, ax = plt.subplots()
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'MMSE', shade=True)
    facetgrid.set(xlim=(0, df['MMSE'].max()))
    facetgrid.add_legend()
    plt.xlim(16.00)
    st.pyplot(facetgrid.fig)
    
    # Dementia Distribution by ASF
    st.subheader('Dementia Distribution by ASF :')
    fig, ax = plt.subplots()
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'ASF', shade=True)
    facetgrid.set(xlim=(0, df['ASF'].max()))
    facetgrid.add_legend()
    plt.xlim(0.6, 1.8)
    st.pyplot(facetgrid.fig)
    
    # Dementia Distribution by ETIV
    st.subheader('Dementia Distribution by ETIV :')
    fig, ax = plt.subplots()
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'eTIV', shade=True)
    facetgrid.set(xlim=(0, df['eTIV'].max()))
    facetgrid.add_legend()
    plt.xlim(900, 2200)
    st.pyplot(facetgrid.fig)
    
    # Dementia Distribution by nWBV
    st.subheader('Dementia Distribution by nWBV :')
    fig, ax = plt.subplots()
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'nWBV', shade=True)
    facetgrid.set(xlim=(0, df['nWBV'].max()))
    facetgrid.add_legend()
    plt.xlim(0.6, 0.9)
    st.pyplot(facetgrid.fig)
    
    # Dementia Distribution by AGE
    st.subheader('Dementia Distribution by AGE :')
    fig, ax = plt.subplots()
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'Age', shade=True)
    facetgrid.set(xlim=(0, df['Age'].max()))
    facetgrid.add_legend()
    plt.xlim(50, 110)
    st.pyplot(facetgrid.fig)
    
    # Dementia Distribution by YEARS OF EDUCATION
    st.subheader('Dementia Distribution by YEARS OF EDUCATION :')
    fig, ax = plt.subplots()
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'EDUC', shade=True)
    facetgrid.set(xlim=(df['EDUC'].min(), df['EDUC'].max()))
    facetgrid.add_legend()
    plt.ylim(0, 0.16)
    plt.xlim(2, 25)
    st.pyplot(facetgrid.fig)
    
    # Drop rows with missing values
    df = df.dropna(axis=0, how="any")
    
    # Dementia Distribution by YEARS OF EDUCATION AND SES
    st.subheader('Dementia Distribution by YEARS OF EDUCATION AND SES :')
    x = df['EDUC']
    y = df['SES']
    ses_not_null = y[~y.isnull()].index
    x = x[ses_not_null]
    y = y[ses_not_null]
    
    # Trend line
    fig, ax = plt.subplots()
    poly = np.polyfit(x, y, 1)
    pp = np.poly1d(poly)
    ax.plot(x, y, 'go', x, pp(x), "b--")
    ax.set_xlabel('Education Level(EDUC)')
    ax.set_ylabel('Social Economic Status(SES)')
    st.pyplot(fig)
    
    plt.show()


    # ======================================================================

elif choice == 'Prediction':

    st.title("Check your Dementia status...")
    df = pd.read_csv(r"oasis_longitudinal.csv")
    # ================================================
    df = df.loc[df['Visit'] == 1]
    # use first visit data only because of the analysis
    df = df.reset_index(drop=True)
    # reset index after filtering first visit data
    df['M/F'] = df['M/F'].replace(['F', 'M'], [0, 1])
    # Male/Female column
    df['Group'] = df['Group'].replace(['Converted'], ['Demented'])
    # Target variable
    df['Group'] = df['Group'].replace(['Demented', 'Nondemented'], [1, 0])
    # Target variable
    df = df.drop(['MRI ID', 'Visit', 'Hand'], axis=1)


    def bar(feature):
        Demented = df[df['Group'] == 1][feature].value_counts()
        Nondemented = df[df['Group'] == 0][feature].value_counts()
        _bar = pd.DataFrame([Demented, Nondemented])
        _bar.index = ['Demented', 'Nondemented']
        _bar.plot(kind='bar', stacked=True, figsize=(8, 5))


    # Gender  and  Group ( Female=0, Male=1)
    bar('M/F')
    plt.xlabel('Group')
    plt.ylabel('Number of patients')
    plt.legend()
    plt.title('Gender v/s Demented rate')
    # =================================================================
    # Create a bar chart using the value_counts() method on the 'M/F' column of the DataFrame
    dementia_by_gender = df[df['Group'] == 1]['M/F'].value_counts()
    dementia_by_gender.plot(kind='bar')
    # Set the title and axis labels
    plt.title('Dementia Distribution by Gender')
    plt.xlabel('Gender (Female=0, Male=1)')
    plt.ylabel('Number of Patients')
    # Display the chart in Streamlit
    # st.pyplot()
    # =====================================================================
    # MMSE : Mini Mental State Examination
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'MMSE', shade=True)
    facetgrid.set(xlim=(0, df['MMSE'].max()))
    facetgrid.add_legend()
    plt.xlim(16.00)
    # st.pyplot()
    # Graph on each variable
    # bar_chart('ASF') = Atlas Scaling Factor
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'ASF', shade=True)
    facetgrid.set(xlim=(0, df['ASF'].max()))
    facetgrid.add_legend()
    plt.xlim(0.6, 1.8)
    # st.pyplot()
    # eTIV = Estimated Total Intracranial Volume
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'eTIV', shade=True)
    facetgrid.set(xlim=(0, df['eTIV'].max()))
    facetgrid.add_legend()
    plt.xlim(900, 2200)
    # st.pyplot()
    # 'nWBV' = Normalized Whole Brain Volume
    # Nondemented = 0, Demented =1
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'nWBV', shade=True)
    facetgrid.set(xlim=(0, df['nWBV'].max()))
    facetgrid.add_legend()
    plt.xlim(0.6, 0.9)
    # st.pyplot()
    # AGE.
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'Age', shade=True)
    facetgrid.set(xlim=(0, df['Age'].max()))
    facetgrid.add_legend()
    plt.xlim(50, 110)
    # st.pyplot()
    # 'EDUC' = Years of Education
    facetgrid = sns.FacetGrid(df, hue="Group", aspect=3)
    facetgrid.map(sns.kdeplot, 'EDUC', shade=True)
    facetgrid.set(xlim=(df['EDUC'].min(), df['EDUC'].max()))
    facetgrid.add_legend()
    plt.ylim(0, 0.16)
    plt.xlim(2, 25)
    df.isnull().sum()
    df = df.dropna(axis=0, how="any")
    pd.isnull(df).sum()
    df['Group'].value_counts()
    # Draw scatter plot between EDUC and SES
    x = df['EDUC']
    y = df['SES']
    ses_not_null = y[~y.isnull()].index
    x = x[ses_not_null]
    y = y[ses_not_null]
    # Trend line
    poly = np.polyfit(x, y, 1)
    pp = np.poly1d(poly)
    plt.plot(x, y, 'go', x, pp(x), "b--")
    plt.xlabel('Education Level(EDUC)')
    plt.ylabel('Social Economic Status(SES)')
    # st.pyplot()
    # plt.show()

    # ============================================================================================================================================    PREDICTION
    exang = st.sidebar.selectbox('Select Your Algorithm',['Simple Linear Regression',"Logistic Regression","SVM"] )
    
    gender = st.selectbox(
        "Gender",
        ("Female", "Male")
    )
    gender = 1 if gender == "Male" else 2
    age = st.selectbox(
        "Age",
        ('18 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54', '55 to 59', '60 to 64',
         '65 to 69', '70 to 74', '75 to 79', '80 or older')
    )
    if age == "18 to 24":
        age = 1
    elif age == "25 to 29":
        age = 2
    elif age == "30 to 34":
        age = 3
    elif age == "35 to 39":
        age = 4
    elif age == "40 to 44":
        age = 5
    elif age == "45 to 49":
        age = 6
    elif age == "50 to 54":
        age = 7
    elif age == "55 to 59":
        age = 8
    elif age == "60 to 64":
        age = 9
    elif age == "65 to 69":
        age = 10
    elif age == "70 to 74":
        age = 11
    elif age == "80 or older":
        age = 12
    else:
        age = 13
    EDUC = st.slider("Years of Education", max_value=30)
    MMSE = st.slider("MMSE Value", max_value=40)
    SES = st.slider("SES Value", max_value=10)
    eTIV = st.slider("eTIV Value", max_value=2040)
    nWBV = st.number_input("nWBV Value")
    ASF = st.number_input("ASF Value")
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.impute import SimpleImputer

    a = pd.read_csv(r"oasis_longitudinal.csv")
    a.isna().sum()
    # create the object of the imputer for null ratings
    im = SimpleImputer(strategy='mean')
    # im= SimpleImputer(strategy= 'most_frequent')
    # fit the ratings imputer with the data and transform
    im.fit(a[['MMSE']])
    a[['MMSE']] = im.transform(a[['MMSE']])
    # create the object of the imputer for null ratings
    im = SimpleImputer(strategy='mean')
    # im= SimpleImputer(strategy= 'most_frequent')
    # fit the ratings imputer with the data and transform
    im.fit(a[['SES']])
    a[['SES']] = im.transform(a[['SES']])
    x = a.iloc[:, 7:-1].values
    y = a.iloc[:, -1].values
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1 / 3, random_state=0)
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    asf = regressor.predict([[87, 13, 2.0, 24, 0.5, 2000, 0.736]])  # for asf
    print(asf)
    # if asf>:
    #    print("Non Dementiated")
    # else:
    #    print("Dementiated")
    x1 = a.iloc[:, 11:14].values
    y1 = a.iloc[:, 10:11].values
    from sklearn.model_selection import train_test_split

    X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=1 / 3, random_state=0)
    from sklearn.linear_model import LinearRegression

    regressor = LinearRegression()
    regressor.fit(X_train1, y_train1)
    y_pred1 = regressor.predict(X_test1)
    mmse = regressor.predict([[nWBV, eTIV, SES]])  # for mmse
    print(mmse)
    print(nWBV, eTIV, SES)
    if st.button("Predict"):
        _, idx = kdtree.query([MMSE, eTIV, nWBV, ASF])
        
        group = df.iloc[idx]['Group']
        #st.success(f"The predicted group is: **{group}**")
        if group=="Demented":
            re = "Dementiated"
            st.warning('Your results indicate possible signs of Alzheimer\'s. ', icon="⚠️")
            st.info("Remember, early detection is key, and there are steps you can take moving forward.")
            st.info("Engage in brain exercises like puzzles, reading, or memory games.")
            st.info("Adopt a Mediterranean diet rich in fruits, vegetables, and healthy fats.")
            st.info("Incorporate regular physical exercise to improve blood flow to the brain.")
            st.info("Maintain a consistent sleep schedule to improve memory and cognitive stability.")
            st.info("Early consultation with a medical expert can help manage the condition effectively")            # Keep the notification displayed for 5 seconds
        else :
            st.success('Your results look good ! No signs of Alzheimer\'s detected.', icon="✅")
            st.info("Remember, a healthy mind thrives with good habits and positivity.")
            
            re = "Non Dementiated"
            
# ============================================================================================
elif choice == "About Us":
    st.info("CREATED BY KARTHICK")

