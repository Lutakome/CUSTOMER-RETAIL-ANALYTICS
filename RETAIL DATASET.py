# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 02:59:08 2024

@author: acer
"""

##CONTEXT OF THE RETAIL ANALYSIS##
#The niche to find the correlation on how different customers are purchasing products is very crucial. Retail businesses, from supermarkets to convenience stores, are constantly seeking ways to better understand their customers and improve their operations. The gap in recommending how customers are linked in a way that if a customer is buying a certain product,what is the affliation with the different products is an interesting question to answer.

#Market basket analysis: Is a data mining technique used in retail analytics, explores customer purchase patterns to uncover associations between products, identify trends, and optimize pricing and promotions. It works by looking for combinations of items that occur frequently in transactions. For example, if customers are buying milk, how probably are they to also buy bread (and which kind of bread) on the same trip to the supermarket? This information may lead to an increase in sales by helping retailers to do selective marketing based on predictions, cross-selling and planning their ledge space for optimal product.

#Customer segmentation: Is the practice of categorizing customers into distinct groups based on characteristics such as; demographics, behavior and preferences.

#Pricing Optimization: Retail businesses are optimizing price strategies and identifying opportunities for discounts and promotions. Identification of optimal price point for any given product at any given location that will yield the highest profit.

##MOTIVATION##
#The motivation for this dataset comes from the need for accessible and customizable market basket datasets. While real-world retail data is sensitive and often restricted, synthetic datasets offer a safe and versatile alternative. Researchers, data scientists, and analysts can use this dataset to develop and test algorithms, models, and analytical tools.

##Use Cases:##
##Market Basket Analysis: Discover associations between products and uncover buying patterns.

##Customer Segmentation: Group customers based on purchasing behavior.

##Pricing Optimization: Optimize pricing strategies and identify opportunities for discounts and promotions.

##Retail Analytics: Analyze store performance and customer trends.

##ABOUT THE DATASET##
##This dataset was created to simulate a market basket dataset, providing insights into customer purchasing behavior and store operations. The dataset facilitates market basket analysis, customer segmentation, and other retail analytics tasks.

##DATASET INFORMATION:##
##The columns provide information about the transactions, customers, products, and purchasing behavior, making the dataset suitable for various analysis, including market basket analysis and customer segmentation. Here's a brief explanation of each column in the Dataset:

##Transaction_ID: A unique identifier for each transaction, represented as a 10-digit number. This column is used to uniquely identify each purchase.

##Date: The date and time when the transaction occurred. It records the timestamp of each purchase.

##Customer_Name: The name of the customer who made the purchase. It provides information about the customer's identity.

##Product: A list of products purchased in the transaction. It includes the names of the products bought.

##Total_Items: The total number of items purchased in the transaction. It represents the quantity of products bought.

##Total_Cost: The total cost of the purchase, in currency. It represents the financial value of the transaction.

##Payment_Method: The method used for payment in the transaction, such as credit card, debit card, cash, or mobile payment.

##City: The city where the purchase took place. It indicates the location of the transaction.

##Store_Type: The type of store where the purchase was made, such as a supermarket, convenience store, department store, etc.

##Discount_Applied: A binary indicator (True/False) representing whether a discount was applied to the transaction.

##Customer_Category: A category representing the customer's background or age group.

##Season: The season in which the purchase occurred, such as spring, summer, fall, or winter.

##Promotion: The type of promotion applied to the transaction, such as "None," "BOGO (Buy One Get One)," or "Discount on Selected Items."




##IMPORTING ALL THE NECESSARY LIBRABRIES##
#Importing libraries in Python is crucial as it enhances your programming capabilities, 
#saves development time, ensures code reliability, and allows you to leverage the extensive 
##work done by the Python community. Libraries are fundamental to the efficiency, modularity, 
##and scalability of your code, enabling you to build robust applications and perform complex tasks with ease.

##IMPORTING ALL LIBRABRIES
import numpy as np ##NumPy is used for numerical computations and array operations.
import pandas as pd## Pandas is used for data manipulation and analysis.
import matplotlib as mlp
import matplotlib.pyplot as plt ## For creating static, interactive, and animated visualizations.
import warnings  # Import Warnings to suppress unnecessary warnings
warnings.filterwarnings("ignore") # Suppress warning messages
import seaborn as sns
import statistics
from tqdm.auto import tqdm
from tabulate import tabulate
import gender_guesser.detector as gender
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
# Import CountVectorizer from sklearn.feature_extraction.text for text feature extraction
from sklearn.feature_extraction.text import CountVectorizer

# Import ast and re for working with text and regular expressions
import ast
import re
import plotly.express as px
import time

##Splitting the Dataset into train and test 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold

# Importing Models classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB

## Importing Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import RocCurveDisplay

###LOADING THE DATA FILE AND IMPORTING FROM A SPECIFIC LOCATION IT INTO PYTHON (DATA PRE-PROCESSING)
#When you first load a dataset in Python, it's essential to carry out 
#several preliminary steps to understand its structure, content, and 
#quality.

#NB: Its a csv file and it is very important to know and identify the 
#location of the file

data = pd.read_csv (r'C:\Users\acer\Documents\JN\Retail_Transactions_Dataset.csv')
### FINISH THE LOADING OF THE DATASET

##INSPECTING THE DATA

##2.1. Check the first few rows:
#Use the head() method to see the first few rows of the dataset, 
##within the parathensis, one can specify the number of first rows 
#they want to view. This also gives a snippet of what the data looks 
#like and also confirming if the right dataset has been loaded or read for use.
data.head(5)
##Close of the first rows ##

##2.2 CHECK THE LAST FEW ROWS

##Similarly, use the tail() method to see the last few rows.
data.tail(5)
## CLOSE OF THE LAST ROWS 

##2.3 GET A CONCISE SUMMARY OF THE DATA FRAME

#The info() method provides a summary of the DataFrame, including the 
##number of non-null entries and data types for each column.
data.info()
##CLOSE OF INFORMATION ON THE DATASET

##2.4 STATISTICAL SUMMARY

##The describe() method provides summary statistics for numerical 
##columns. This foundational analysis is crucial for understanding 
##the distribution and central values in your data.
data.describe()
##close of the description 

#2.5 CHECKING THE DIMENSIONS IN THE DATASET

##Shape is used to retrieve the dimensions of a dataset. 
#It provides a tuple representing the number of rows and columns in 
#the dataset. This information is crucial for understanding the 
##structure of your data, performing data validation, and preparing 
##data for further analysis or modeling
data.shape
## We have 1000000 columns and 13 rows

#Understand Data Types and Missing Values

#3.1. Data types: Check the data types of each column.
print(data.dtypes)
##close of different types with;int, object, float and boolean

##3.2 Missing values
##Check for missing values in the dataset.
print(data.isnull().sum())
##close of checking for missing values with their total count. 
## Promotion colimn has missing values of 333943 which is 33% of the 
##promotion column


## 4 DATA CLEANING/ CLEANSING
## HANDLING MISSING VALUES

##We can see that the death_rate column has missing values with a total sum of 333943 counts They are various ways to decide how to handle missing values: drop, fill, or other imputation methods.

#1. Drop/Delete the missing values is not efficient/ suitable if 
#the dataset is big.
##2. The imputation method is when we find the mean/mode value for 
##that column in the dataset and then replace it with the same value.
##3. Forward fill or backward fill: Use the adjacent values to fill 
##missing entries.
##4. Fill the missing values with specific words like 'unknown'.
##5. Delete the rows with the missing the values.
#NB: In this dataset, we are going to use the fourth option as the 
##missing values are many.
##This is also because the Promotion column is of object data type
##and does not have a statistical summary of mean, mode or standard deviation.
## 4.1 Dropping Columns
data.drop(['Transaction_ID'],axis=1,inplace=True)

## 4.2 Checking columns with missing values with the specific count or total sum
print(data.isnull().sum())##

### 4.3 Using a specific word to fill the rows with null values
data['Promotion'].fillna('unknown', inplace=True)
print(data)
###close for filler

## 4.4 Checking for the data types in the dataset. 
###N.B Data Types also tend to drive the flow of the dataset, conversion of data types is very necessary. 
##Visualiations are more swift with 'int' and 'float' data types
print(data.dtypes)
###close 

##4.5 Presenting variables, dtypes, count, unique and missing values
#in a table format

##Also confirming if the missing values have been dropped.
def describe(data):
    
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []
    min_ = []
    max_ = []
    
    for item in data.columns:
        variables.append(item)
        dtypes.append(data[item].dtype)
        count.append(len(data[item]))
        unique.append(len(data[item].unique()))
        missing.append(data[item].isna().sum())

        if data[item].dtypes == 'float64' or data[item].dtypes == 'int64':
            min_.append(data[item].min())
            max_.append(data[item].max())
        else: 
            min_.append('Str')
            max_.append('Str')
            
    output = pd.DataFrame({
        'variable': variables,
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing values' : missing,
        'Min': min_,
        'Max': max_
    })
    
    return output

## 4.5.1 Creating a Dictionary for the Table Outlook
Styles = [dict(selector = 'caption',
              props = [('color', 'darkslategray'),
                      ('text-align', 'center'),
                      ('font-size', '15pt')])]
### 4.5.2 Presenting the final outlook of the Table
df_ = describe(data)
df_ = df_.style.set_caption('Table 1 : overview of dataset').set_table_styles(Styles)
df_
##close it off##

#4.6 Extracting Year, Month and Day from the date variable

#calendar.month_name: Provides a sequence of month names with an empty string at the start.
#list(calendar.month_name): Converts this sequence to a list.
#[1:]: Slices the list to exclude the first element, resulting in a 
#list of month names from January to December. By using this line of code, 
##you can easily generate a list of month names without the initial empty string, which can be useful for various applications, such as labeling plots, 
##creating dropdown menus, or any other functionality requiring month names.
import calendar
months=list(calendar.month_name)[1:]
##close

##4.7 CHECKING FOR DUPLICATE VALUES

##Checking for duplicate values is a critical step in data preprocessing.
##It helps maintain data integrity, ensures statistical accuracy, improves model performance, 
##and enhances resource efficiency
# Check for duplicate rows
duplicate_rows = data[data.duplicated()]

# Print duplicate rows if any
if not duplicate_rows.empty:
    print("Duplicate rows found:")
    print(duplicate_rows)
else:
    print("No duplicate rows found")

# Drop duplicate rows if any
if not duplicate_rows.empty:
    data.drop_duplicates(inplace=True)
    print("Duplicate rows dropped")
else:
    print("No duplicate rows were dropped")
 ##close off

 # Function to extract the first name
def get_first_name(full_name):
     return full_name.split()[0]

 # Apply the function to the 'FullName' column to create a new 'FirstName' column
data['FirstName'] = data['Customer_Name'].apply(get_first_name)

 # Drop the 'FullName' column if you only want to keep the first names
data.drop(columns=['Customer_Name'], inplace=True)

 # Print the DataFrame to verify the results
print(data.head())

 # Save the updated DataFrame to a new CSV file
data.to_csv('data_with_first_names.csv', index=False)

 # Initialize the gender detector
d = gender.Detector()

 # Function to detect gender
def detect_gender(name):
     return d.get_gender(name)

 # Apply the gender detection function to the 'Name' column and create a new column 'Gender'
data['Gender'] = data['FirstName'].apply(detect_gender)

 # Print the DataFrame to verify the results
print(data.head(10))

 # Save the updated DataFrame to a new CSV file
data.to_csv('data_with_gender.csv', index=False)

data.Gender.value_counts()

 # Replace 'mostly_female' with 'female' in the 'Gender' column
data['Gender'] = data['Gender'].replace('mostly_female', 'female')
 # Replace 'mostly_male' with 'male' in the 'Gender' column
data['Gender'] = data['Gender'].replace('mostly_male', 'male')
 # Replace 'andy' with 'female' in the 'Gender' column
data['Gender'] = data['Gender'].replace('andy', 'female')

data.loc[(data['Gender'] == 'unknown') & (data['FirstName'].str.contains('Mrs.')), 'Gender'] = 'female'
data.loc[(data['Gender'] == 'unknown') & (data['FirstName'].str.contains('Mr.')), 'Gender'] = 'male'
 # Replace 'mostly_female' with 'female' in the 'Gender' column
data['Gender'] = data['Gender'].replace('unknown', 'female')
data.Gender.value_counts()

 # Save the updated DataFrame to a new CSV file
data.to_csv(r'C:\Users\acer\Documents\JN\RetaiR_Transactions.csv', index=False)


 # Remove square brackets and single quotes from the 'product' column
data['Product'] = data['Product'].str.replace('[', '').str.replace(']', '').str.replace("'", '')

 # Print the DataFrame to verify the changes
print(data.head(5))
print(data.dtypes)
print(data.shape)
 # Save the updated DataFrame to a new CSV file
data.to_csv('data_with_gender.csv', index=False)  

data.drop(['FirstName'],axis=1,inplace=True)
print(data.dtypes)
 
   ## 5.0 EXPLORATORY DATA ANALYSIS##
#Exploratory Data Analysis (EDA) is a critical step in the data analysis process. 
#It involves summarizing and visualizing the main characteristics of a dataset, often with the help of graphical representations. Analysis can be done in three ways; Univariate, Bi-Variate and Multi-Variate

#Univariate Analysis : Analyze individual columns to show distribution,
#central tendancy and dispersion. The possible plots include; Histograms, Box Plots, Value Counts. 
##Bi-Variate Analysis: Analyze relationships between pairs of variables. 
##The possible plots include; Scatter Plots, Correlation Matrix and Heat Map, Box Plot (Continous vs Categorical Data) 
##Mulit-Variate Analysis: Analyze relationships between more than 2 variables. 
##The possible plots include; Pair Plots, Facet Plots, Swarm Plots. 
##NB: Different Visualizations will take place to check how different paramaters flow through.
from matplotlib import cm
c1=cm.get_cmap('Set1')
## close off

## 5.1 Creating a Bar Plot to check the 10 top most product which is the most purchased
## Interpreation: Tooth Paste is the most purchased product for all the products
plt.figure(figsize=(10,5))
top_10_products = data['Product'].value_counts().nlargest(10)
ax = sns.barplot(y=top_10_products.index,x=top_10_products.values)
for bars in ax.containers:
    ax.bar_label(bars)

## 5.2 Checking how customers tend to purchase certain products according to certain seasons
newdfs=data[['Product','Season']].groupby('Season') 

#5.2.1 Extracting the unique values from a specific column in a pandas DataFrame. Here's a breakdown of what this code does:
#Explanation¶
#data.Season:data is a pandas DataFrame.
#Season is the name of the column in the DataFrame from which you want to get the unique values.
#unique():The .unique() method is called on the Season column. 
#This method returns the unique values in that column as a numpy array.
seasons=data.Season.unique()   
## close off 


#5.2.2 A Function converter that takes a string as input and removes 
#specific characters from it.
def converter(input_element):
    return input_element.replace('[','').replace(']','').replace("'","")
## close 

##5.2.3 The provided code processes each element in the answer list 
#to create a DataFrame for each season. Each DataFrame contains the 
#product names and their respective counts, and the index name is set
#to the season. These DataFrames are then stored in a list called dataframe.
dfs=[]
for i in seasons:
    dfs.append((i,newdfs.get_group(i)['Product'].apply(converter)))
##close off

##5.2.4 The code consists of a function productcount and a loop that 
##applies this function to a list dfs
def productcount(season,df):
    
    k={}
    for i in df:
        for j in i.split(','):
            j=j.strip()
            if j not in k:
                k[j]=1
            else:
                k[j]+=1
    return season,k 

answer=[]
for i in range(len(dfs)):
    answer.append(productcount(dfs[i][0],dfs[i][1]))
    ## close off
  
##5.2.5
dataframe=[]
for i in range(len(answer)):
    df=pd.DataFrame(dict(products=answer[i][1].keys(),count=answer[i][1].values()))
    df.index.name=answer[i][0]
    dataframe.append(df)
##close off

#5.2.6
winter=dataframe[0].sort_values(by='count',ascending=False).head(5)
Fall=dataframe[1].sort_values(by='count',ascending=False).head(5)
summer=dataframe[2].sort_values(by='count',ascending=False).head(5)
spring=dataframe[3].sort_values(by='count',ascending=False).head(5)
##close off

##5.2.6
fig,axes=plt.subplots(2,2,figsize=(15,15))

sns.barplot(x=winter.products,y=winter['count'],ax=axes[0][0],palette='Set1')
sns.barplot(x=Fall.products,y=Fall['count'],ax=axes[0][1],palette='hsv')
sns.barplot(x=summer.products,y=summer['count'],ax=axes[1][0],palette='Set2')
sns.barplot(x=spring.products,y=spring['count'],ax=axes[1][1],palette='Set3')
title=["Winter","Fall","Summer","Spring"]
count=0
for i in range(2):
  for j in range(2):
    axes[i][j].set_xticklabels(axes[i][j].get_xticklabels(),rotation=20)
    axes[i][j].set_title(title[count])
    axes[i][j].set_ylabel('Frequency')
    count+=1
plt.suptitle('Top 5 Products in Each Season')
plt.tight_layout()
##close off 

##5.3.1
sample1=[]
for i in data.Product:
  sample=i[1:-1].split(',')
  for j in sample:
    sample1.append(j.strip())
products=set(sample1)
##close off

#5.3.2
yval=pd.Series(sample1).value_counts()
xval=yval.index
fig=plt.figure(figsize=(20,20))
ax=fig.add_axes([1,1,1,1])
sns.barplot(x=xval,y=yval,ax=ax,palette='hsv')
plt.xticks(rotation=90,fontsize=15)
plt.title('Total Products Frequency')

##5.4.1 
data.Date=pd.to_datetime(data.Date)
year=[]
month=[]
for i in data.Date:
  year.append(i.strftime("%Y"))
  month.append(i.strftime("%m"))
##close off

##5.4.2
#data.drop("Date",axis=1,inplace=True)
## close off

##5.4.3 
data["year"]=year
data['month']=month
data['year']=data['year'].astype('int')
data.month=data['month'].astype('int')

year1=list(data.groupby("year"))[0][1]
year2=list(data.groupby("year"))[1][1]
year3=list(data.groupby("year"))[2][1]
year4=list(data.groupby("year"))[3][1]
##close off

##5.4.4 
year1cost=[]
year2cost=[]
year3cost=[]
year4cost=[]
for i in range(1,13):
  st='month == {}'.format(i)
  year1cost.append(year1.query(st).Total_Cost.sum())
  year2cost.append(year2.query(st).Total_Cost.sum())
  year3cost.append(year3.query(st).Total_Cost.sum())
  year4cost.append(year4.query(st).Total_Cost.sum())
## close off 

##5.4.5
fig,axes=plt.subplots(2,2,figsize=(15,15))

sns.lineplot(x=months,y=year1cost,ax=axes[0][0],color='green')
sns.lineplot(x=months,y=year2cost,ax=axes[0][1],color='red')
sns.lineplot(x=months,y=year3cost,ax=axes[1][0],color='cyan')
sns.lineplot(x=months,y=year4cost,ax=axes[1][1],color='purple')
title=["2020","2021","2022","2023"]
count=0
for i in range(2):
  for j in range(2):
    axes[i][j].set_xticklabels(axes[i][j].get_xticklabels(),rotation=20)
    axes[i][j].set_title(title[count])
    count+=1
plt.suptitle('Total Sales in Each Years')
plt.tight_layout()
##close off 

##5.5
plt.figure(figsize=(10,5))
chart = data.groupby(['Payment_Method'],as_index=False)['Total_Items'].sum().sort_values(by='Total_Items',ascending=False)
sns.barplot(data=chart, y='Payment_Method',x='Total_Items')
##close off 

##5.6.1  UNIVARIATE ANALYSIS 
categorical_cols_to_plot = data.select_dtypes(include=['object','category']).columns.to_list()
# We select the numerical variables.
numerical_cols_to_plot = data.select_dtypes(include=['int','float']).columns.to_list()
## CLOSE OFF

#5.6.2
def shapiro_test(data:pd.DataFrame, col:str):
    stat, p_value = shapiro(data[col])
    if p_value < 0.05:
        return p_value, 'No Normal Ditribution'
    else:
        return p_value, 'Normal Distribution'
## CLOSE OFF

##5.6.3 
def univariate_numerical_plot(data:pd.DataFrame, var:str):
  """
  Args:
    - data(pd.DataFrame): data.
    - var(str):variable a trazar.
  """
  ax = plt.figure(constrained_layout = False, figsize = (12,5.8)).subplot_mosaic("""AD
                                                                                 BD""")
  sns.boxenplot(data, x= var, ax = ax['A'], color = 'lime')
  sns.stripplot(data, x = var, alpha = 0.5, color = 'darkblue', ax = ax['A'])
  sns.histplot(data, x = var, kde = True,line_kws = {'linewidth':1.8}, color = '#FF5733', ax = ax['B'])
  qqplot(data[var], line = 's', ax = ax['D'])
  df_info = data[var].describe()
  ax['A'].set_xlabel('')
  ax['A'].set_title(f'Mean={round(df_info[1],2)} | Std={round(df_info[2],2)} | Median={round(df_info[5],2)}', fontsize = 9, fontweight='bold')
  ax['B'].set_xlabel('')
  ax['D'].set_title(f'QQ-Plot | Shapiro test: p-value={round(shapiro_test(data,var)[0],3)} | {shapiro_test(data,var)[1]}',fontsize=9, fontweight='bold')
  plt.suptitle(f'Distribution of variable {var}',fontsize = 14, fontweight = 'bold', color = 'darkred')
  plt.tight_layout()
  plt.subplots_adjust(top=0.9)
  plt.show()
## CLOSE OFF

##5.6.4
import matplotlib.pyplot as plt
from scipy.stats import shapiro
plt.style.use('ggplot')
from scipy.stats import shapiro
from statsmodels.graphics.gofplots import qqplot
import re
%matplotlib inline
univariate_numerical_plot(data, numerical_cols_to_plot[0])
##CLOSE OFF

##5.7.1
import matplotlib.pyplot as plt
%matplotlib inline
univariate_numerical_plot(data, numerical_cols_to_plot[1])
##CLOSE OFF 

##5.8 MULTIVARIATE ANALYSIS 
def num_vs_cat_variables(data:pd.DataFrame, var_num:str, var_cat:str):

    ax = plt.figure(constrained_layout = False, figsize = (15, 6.2)).subplot_mosaic("""AD
                                                                               BD""")
    sns.boxplot(data, y = var_num, ax = ax['D'])
    sns.stripplot(data, y = var_num, alpha = 0.2, hue = var_cat, ax = ax['D'])
    sns.histplot(data, x = var_num, hue = var_cat, line_kws = {'linewidth':1.8}, ax = ax['B'])
    sns.kdeplot(data, x = var_num, hue = var_cat, ax = ax['A'])
    ax['A'].set_xlabel('')
    ax['B'].set_xlabel('')
    plt.suptitle(f'Distribution of variable {var_num} by {var_cat}',fontsize = 14, fontweight = 'bold', color = 'darkred')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
##CLOSE OFF

#import matplotlib.pyplot as plt
#%matplotlib inline
#num_vs_cat_variables(data, numerical_cols_to_plot[1])
#CLOSE OFF

### FLATTEN ALL STRINGS IN THE PRODUCTS COLUMN
class ProductMonth:

  def __init__(self, dataframe):
      self.data = dataframe
      self.original_data = dataframe
      self.transform_products_to_flatten_list()

  def preprocess_product_by_month(self, row) -> None:
    """
    Preprocesses the product column for a given row by converting the 
    product strings to lists and flattening them.
    """
    self.data['Product'].iloc[row] = [eval(i) for i in self.data['Product'].iloc[row]]
    self.data['Product'].iloc[row] = [i for i in flatten(self.data['Product'].iloc[row])]
        
  def transform_products_to_flatten_list(self) -> None:
    """
    Transforms the products in the dataframe to a flattened list.
    """
    for month_row in np.arange(self.data.shape[0]):
        self.preprocess_product_by_month(row=month_row)

   def get_product_total_by_month(self, month_index: int, normalize: bool):
       """
       Returns the total count of each product for the specified month.
       """
       index_loc = month_index - 1
       return pd.Series(self.df.iloc[index_loc]['product'])\
            .value_counts(normalize=normalize)

   def get_mm_volume_change(self, prior: int, current: int):
       """
       Returns the month to month volume change of a product.
       """
       mm_change_data = (
           self.get_product_total_by_month(prior, normalize=True)\
              - self.get_product_total_by_month(current, normalize=True))\
          .to_frame().round(4) * 100
       return mm_change_data


# aggregate first before adding into pipeline
Product_data_by_month = retail_df_new\
    .groupby(['date_month'])\
    .agg({'product': lambda x: list(x)})\
    .reset_index()

# add aggregated data into pipeline
#product_month_pipe = ProductMonth(dataframe=product_df_by_month)
##6.0 CONVERTING DATA TYPES 
# Convert the 'gender' column to categorical dtype
data['Gender'] = data['Gender'].astype('category')
print(data['Gender'].dtype)  # Output: category

# Encode categorical variables. We use label encoding 
# Initialize the LabelEncoder
label_encoder = LabelEncoder()
data['Payment_Method'] = label_encoder.fit_transform(data['Payment_Method'])
data['Customer_Category'] = label_encoder.fit_transform(data['Customer_Category'])
data['Season'] = label_encoder.fit_transform(data['Season'])
data['Store_Type'] = label_encoder.fit_transform(data['Store_Type'])
data['Promotion'] = label_encoder.fit_transform(data['Promotion'])

## CLOSE OFF
# Encode the target variable
data['gender_encoded'] = label_encoder.fit_transform(data['Gender'])

# Check the encoded values
print(data[['Gender', 'gender_encoded']])

#6.1 CONVERTING BOOLEAN TO AN INTGER WHERE 1 = TRUE AND 0 = False
data['Discount_Applied'] = data['Discount_Applied'].astype(int)
##CLOSE OFF

## 7.0 CORRELATION MATRIX 
# Exclude non-numeric columns
numeric_data = data.select_dtypes(include=[float, int])

# Calculate the correlation matrix for numeric columns only
numeric_correlation_matrix = numeric_data.corr()

print("Numeric Correlation Matrix:")
print(numeric_correlation_matrix)
## CLOSE OFF

## 7.1 HEAT MAP
# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
##CLOSE OFF

#8.1 ANOVA ANALYSIS
#When we have a categorical output feature and a numerical input 
#feature, we can use the Analysis of Variance (ANOVA) test to check 
#the association between the features. ANOVA is a statistical method 
#used to analyze the differences among group means in a sample. 
#It can help determine whether there are statistically significant 
#differences in the means of the numerical input feature across different 
#categories of the categorical output feature.
from scipy.stats import f_oneway

def check_feature_association_through_ANOVA_test(data, input_feature_list, output_feature):
  association_featues = []
  for i in input_feature_list:
      # Split data based on unique values of the output feature
      output_categories = data[output_feature].unique()
      data_groups = [data[data[output_feature] == category][i] for category in output_categories]

      # Perform ANOVA test
      statistic, p_value = f_oneway(*data_groups)


      # Check if the p-value is below your significance level (e.g., 0.05) to determine significance
      if p_value < 0.05:
          association_featues.append(i)
          print(f"✅ There is a significant association between the features ({i} and {output_feature}).")
      else:
          print()
          print(f"❌ There is no significant association between the features ({i} and {output_feature}).")

      # Print the results
      print(f"ANOVA Statistic: {statistic}")
      print(f"P-value: {p_value}")
      print()

  return association_featues
##CLOSE OFF

##8.1.1 SELECTING THE FEATURES TO USE
features = ['Total_Items', 'Payment_Method', 'Store_Type', 
            'Discount_Applied', 'Season', 'year', 'month']
##CLOSE OFF

##8.1.2 Printing the ANOVA ANALYSIS
#association_featues = check_feature_association_through_ANOVA_test
(data, features, 'Total_Cost')
##CLOSE OFF




##9.1 ASSOCIATION ANALYSIS ##
# Create a one-hot encoded DataFrame
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(data['Product']).transform(data['Product'])
data_onehot = pd.DataFrame(te_ary, columns=te.columns_)

print("One-Hot Encoded DataFrame:")
print(data_onehot)
## CLOSE OFF 

##9.2 USING THE APRIORI ALGORITHM (SUPPORT, LIFT AND CONFIDENCE)
from mlxtend.frequent_patterns import apriori

# Find frequent itemsets with a minimum support of 0.6
frequent_itemsets = apriori(data_onehot, min_support=0.6, use_colnames=True)

print("Frequent Itemsets:")
print(frequent_itemsets)
##CLOSE OFF

##9.3 SETTING ASSOCIATION RULES##
from mlxtend.frequent_patterns import association_rules

# Generate association rules with a minimum confidence of 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Association Rules:")
print(rules)
##CLOSE OFF 

##10.1 MODELLING ##
##SPLIT TRAIN TEST
#The purpose of this split is to allow you to train your model on one 
#set of data and then evaluate its performance on another set of data 
#that it hasn't seen before.

#The most common split is a 70/30 or 80/20 split, where the larger 
#portion is used for training the model and the smaller portion is 
#used for evaluating its performance. The training set is used to fit 
#the model to the data, while the testing set is used to evaluate the 
#model's performance in making predictions on new, unseen data.

#So, we decide to split the dataset 70% for training and 30% for testing.
###Splitting the Data into Training and Testing Sets
###The train_test_split function is imported from the scikit-learn library

from sklearn.model_selection import train_test_split 

# Define features and target and selecting the feature and the target variables
##We can tend to have so many features but the target column is very specific
X = data[['Season', 'Discount_Applied', 'Promotion']]  # Feature 
y = data['gender_encoded']  # Target variable

# Split the data into training and testing sets
###Test Size is a paramt er to change the proportion of the data used for testing as it is declared with a choosen split 
###Random State ensures the spilit is reproducible, this can be any integer
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.info())
print(X_test.info())
# Verify the split
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
##CLOSE OFF

##10.2 Normalize the Data
#Normalization, in the context of data analysis and machine learning,
# refers to the process of transforming numerical data into a standardized scale. 
#The goal of normalization is to bring different features or variables 
#onto a similar scale to ensure fair comparisons and improve the performance 
#of certain algorithms.
# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_features = ['Total_Cost', 'Total_Items']
data[numerical_features] = scaler.fit_transform(data[numerical_features])
##CLOSE OFF

##10.2.1 
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)
##CLOSE OFF

##10.2.2 
X_train_std = pd.DataFrame(X_train_std, columns = X_train.columns)
X_test_std = pd.DataFrame(X_test_std, columns = X_train.columns)
##CLOSE OFF

##10.3 INTIALIZING MODELS  
#models = {
#    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
#    'DecisionTree': DecisionTreeClassifier(random_state=42),
#    'LGBM': LGBMClassifier(random_state=42),
#    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
#    'NaiveBayes': GaussianNB(),
#    'SVC': SVC(probability=True, random_state=42),
#    'XGB': XGBClassifier(random_state=42)
#}

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Elastic Net': ElasticNet(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Support Vector Regressor': SVR()
}


##10.3.1 MODEL PERFORMANCE 

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    print(f"{name}: MSE = {mse}, RMSE = {rmse}, MAE = {mae}, R² = {r2}")
# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print(results_df)

results_df[['MSE', 'RMSE', 'MAE']].plot(kind='bar', figsize=(14, 8))
plt.title('Model Comparison - Error Metrics')
plt.ylabel('Error')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()

results_df[['R²']].plot(kind='bar', figsize=(14, 8))
plt.title('Model Comparison - R² Score')
plt.ylabel('R²')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.show()
##CLOSE##

## ANALYZING MODEL PERFORMANCE USING VIZUALS ##
plt.figure(figsize=(10, 6))
for name, model in models.items():
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.5, label=name)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual Gender')
plt.ylabel('Predicted Gender')
plt.title('Actual vs Predicted Gender')
plt.legend()
plt.show()
## CLOSE

## RESIDUAL PLOTS ##
plt.figure(figsize=(14, 8))
for name, model in models.items():
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, label=name, alpha=0.5)

plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.legend()
plt.show()
results = {}

#for name, model in models.items():
    # Train the model
#    model.fit(X_train, y_train)
    
    # Predict on the test set
#    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    # Print classification report
#    print(f"{name} Classification Report:")
#    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
 #   conf_matrix = confusion_matrix(y_test, y_pred)
 #   sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
 #   plt.title(f"{name} Confusion Matrix")
 #  plt.xlabel('Predicted')
  #  plt.ylabel('True')
 #   plt.show()
## CLOSE OFF 


#10.4  Plot model accuracy
#model_names = list(results.keys())
#model_accuracies = list(results.values())

#plt.figure(figsize=(10, 6))
#sns.barplot(x=model_names, y=model_accuracies, palette='viridis')
#plt.title('Model Accuracy Comparison')
#plt.xlabel('Model')
#plt.ylabel('Accuracy')
#plt.ylim(0, 1)
#plt.show()
## CLOSE OFF   