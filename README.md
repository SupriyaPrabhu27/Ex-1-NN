<H3>ENTER YOUR NAME</H3> SUPRIYA PRABHU
<H3>ENTER YOUR REGISTER NO.</H3> 212224240165
<H3>EX. NO.1</H3> 
<H3>DATE</H3> 28-10-2026
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\Users\admin\TRAFFIC_DATA.csv")
df


df.isnull().sum()


df = df.fillna(0)
df.isnull().sum()


df.duplicated()


df = df.drop_duplicates()

# Display statistical information
df.describe()

# Convert categorical columns into numerical values
df = pd.get_dummies(df, drop_first=True, dtype=int)

# Scale numerical columns
inc_cols = df.select_dtypes(include=['number']).columns.tolist()
scaler = StandardScaler()
df[inc_cols] = scaler.fit_transform(df[inc_cols])

df

# Separate input and output
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("X Values")
x

print("Y Values")
y

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print("X Training data")
x_train

print("X Testing data")
x_test

```

## OUTPUT:


<img width="1373" height="512" alt="image" src="https://github.com/user-attachments/assets/8dc11271-dd51-4bd2-aa18-adfc71239fe6" />


<img width="1371" height="508" alt="image" src="https://github.com/user-attachments/assets/64fa5d5e-878f-4ba8-b95a-44189cf96ec7" />

<img width="1391" height="467" alt="image" src="https://github.com/user-attachments/assets/518ac17a-f6aa-439a-8225-c9c101b297d5" />



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


