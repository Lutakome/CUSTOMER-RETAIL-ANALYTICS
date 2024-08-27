# CUSTOMER-RETAIL-ANALYTICS
CRISP-DM LIFECYCLE DOCUMENTATION FOR A RETAIL DATASET
CRISP-DM is a process model that describes common approaches used by data mining experts. It is an open standard and the most widely-used analytics model. This document contains models that describes the lifecycle of a data science project that is intended to solve a retail business problem. This project is broken into 6 distinct phases starting from the inception of the project all the way to its final impact. The project runs on a CRISP-DM (Cross-Industry Standard Process for Data Mining) lifecycle. The key phases for the project (Figure 1): Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation, and Deployment. In the documentation below, you will find detailed information on each of the phases, practical information (code implementation), important considerations, ethical and moral policies to be undertaken. 

1.0 PHASE ONE: BUSINESS UNDERSTANDING
1.1 BACKGROUND OF THE PROJECT (CUSTOMER RETAIL ANALYSIS)
The niche to find the correlation on how different customers are purchasing products is very crucial. Retail businesses, from supermarkets to convenience stores, are constantly seeking ways to better understand their customers and improve their operations. The gap in recommending how customers are linked in a way that if a customer is buying a certain product, what is the affiliation with the different products is an interesting question to answer.
Market basket analysis: Is a data mining technique used in retail analytics, explores customer purchase patterns to uncover associations between products, identify trends, and optimize pricing and promotions. It works by looking for combinations of items that occur frequently in transactions. For example, if customers are buying milk, how probably are they to also buy bread (and which kind of bread) on the same trip to the supermarket? This information may lead to an increase in sales by helping retailers to do selective marketing based on predictions, cross-selling and planning their ledge space for optimal product.
Customer segmentation: Is the practice of categorizing customers into distinct groups based on characteristics such as; demographics, behaviour and preferences.
Pricing Optimization: Retail businesses are optimizing price strategies and identifying opportunities for discounts and promotions. Identification of optimal price point for any given product at any given location that will yield the highest profit.

1.2 MOTIVATION
The motivation for this dataset comes from the need for accessible and customizable market basket datasets. While real-world retail data is sensitive and often restricted, synthetic datasets offer a safe and versatile alternative. Researchers, data scientists, and analysts can use this dataset to develop and test algorithms, models, and analytical tools.

1.3 USE CASES:
Market Basket Analysis: Discover associations between products and uncover buying patterns.
Customer Segmentation: Group customers based on purchasing behaviour.
Pricing Optimization: Optimize pricing strategies and identify opportunities for discounts and promotions.
Retail Analytics: Analyse store performance and customer trends.

1.3.1 Business Context: 
The project is important for the business because they would like to have a better insight that can revolutionize marketing strategies and ultimately enhance customer satisfaction and loyalty.

1.3.2 Success Criteria: 
The success will be measured by increased customer loyalty and revenue growth.

1.4 Business Questions
1.4.1 Primary Question of the Project: 
The primary question of this project is to predict customer Gender based on their shopping behaviours and other relevant features present in the dataset.

1.4.2 Secondary Questions:
What are the most popular products?

Which products are often purchased together?

In which season do the customers tend to purchase most?

How can we identify and target high-value customers?

Which features will influence the customers the most and less when they are in the supermarket?

1.4.3 Project Goals:
Develop a recommendation system to suggest products to customers.

Segment customers to tailor marketing strategies.

Predict sales trends to optimize inventory management.

2.0 PHASE TWO: DATA UNDERSTANDING
Acquiring, storing, linking, understanding data for analysis on a project often entails an involved and iterative process, requiring working closely with the team to work on the choice of the dataset to ensure that the dataset collected and chosen answers the business objectives. Consistency and reliability in the data format was essential to ensure protection of private and sensitive information. During this phase of the work, the team was required to apply skills working with and structuring data to get it into a storage format that is appropriate for linking it with the data sources. Each of the steps require active communication with the project’s stakeholders to understand the context in which the data were collected and structured to ensure data definitions actually describe the events they are supposed to reflect. 

2.1 Data Collection:
2.1.1 Data Source: 
The data was collected through team work. We used Kaggle, which is a platform for finding datasets, participate in data science competitions, collaborate with other data scientists etc.
The title of our dataset is: Retail Transaction Dataset

The link to access this dataset: 
https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset/data
Data Files: Retail Transactions Dataset.csv

2.1.2 Data Acquisition: 
The data was acquired because this was our task during our Data Science internship with London School of Informatics.

2.2 Data Inventory: 
The Retail Transaction Dataset which we have collected was created to simulate a market basket dataset, providing insights into customer purchasing behaviour and store operations. The dataset facilitates market basket analysis, customer segmentation, and other retail analytics tasks.

2.2.1 Initial Data Exploration:
Retail Transactions Dataset.csv: Transaction_ID, Customer Name, Date, Product, Total_Items, Total_Cost, Payment_Method, City, Store_Type, Discount_Applied, Customer_Category, Promotion

2.2.2 Summary Statistics:

Number of transactions/rows: 100,000, 000

Number of columns: 13

2.2.3 Data Quality Issues:

In our dataset we did not have any outliers or duplicates.

Missing values in Promotion column.

Inconsistent customer names.

2.2.4 ABOUT THE DATASET
This dataset was created to simulate a market basket dataset, providing insights into customer purchasing behaviour and store operations. The dataset facilitates market basket analysis, customer segmentation, and other retail analytics tasks.

2.2.4.1 Dataset Information (Attributes and Variables):
The columns provide information about the transactions, customers, products, and purchasing behaviour, making the dataset suitable for various analysis, including market basket analysis and customer segmentation. Here's a brief explanation of each column in the Dataset:

Transaction_ID: A unique identifier for each transaction, represented as a 10-digit number. This column is used to uniquely identify each purchase.

Date: The date and time when the transaction occurred. It records the timestamp of each purchase.

Customer_Name: The name of the customer who made the purchase. It provides information about the customer's identity.

Product: A list of products purchased in the transaction. It includes the names of the products bought.

Total_Items: The total number of items purchased in the transaction. It represents the quantity of products bought.

Total_Cost: The total cost of the purchase, in currency. It represents the financial value of the transaction.

Payment_Method: The method used for payment in the transaction, such as credit card, debit card, cash, or mobile payment.

City: The city where the purchase took place. It indicates the location of the transaction.

Store_Type: The type of store where the purchase was made, such as a supermarket, convenience store, department store, etc.

Discount_Applied: A binary indicator (True/False) representing whether a discount was applied to the transaction.

Customer_Category: A category representing the customer's background or age group.

Season: The season in which the purchase occurred, such as spring, summer, fall, or winter.

Promotion: The type of promotion applied to the transaction, such as "None," "BOGO (Buy One Get One)," or "Discount on Selected Items."

NB: The dataset has 2 numerical columns and more categorical ones.

3.0 PHASE THREE: DATA PREPARATION
At this phase (also be referred to as "data munging"), the data is prepared for modelling. It has the following tasks and more as the data demands.

4.0 PHASE FOUR: MODELLING
4.1 Model Building
Tools and Libraries: The library I used for modelling is scikit-learn.
Model Training and Validation: The data was divided into 2 parts, testing set (20%) and training set (80%). We used the train set to teach the model and test set for the validation of the model.
Our target variable is gender and the features are the rest of the variables in the dataset. This implies that the gender takes up the 20% and the rest of the variables are the 80%. 

4.2 Model Selection
Modelling Techniques: We used different mused K-Nearest Neighbors, Support Vector Classifier, Naive Bayes and Decision Tree Classifier. 
Justification: The used and selected algorithms were the suggested models by the stakeholders (London School Informatics).

4.3 Model Evaluation
Evaluation Metrics: The metrics were used to evaluate model performance are: accuracy, F1 score, precision, recall. One of the crucial tasks in the application of machine learning is evaluating the execution of procedures. A precise model that can forecast the previously unobserved data must be developed. The model might perform well by some metrics, but by others, its strength might perform poorly. Therefore, it is crucial to use a variety of evaluation indicators to rate the model. 


5.0 Evaluation
5.1 Model Performance Metrics
Comparison: The performance is not very good for any of the models, the Linear SVC and Naive Bayes are very poor in precision and F1 Score.
During the analysis, before using the models I used feature scoring. It helps us to understand the importance of different features in our model, it tells us which features have more influence on the predictions. In our project I highlighted with yellow the column names which got the best scores.

Accuracy: High accuracy indicates that the model correctly classifies a high proportion of instances. However, it is not always the best metric for imbalanced datasets.

Precision and Recall: High precision and recall are indicators of a good balance between false positives and false negatives. Retail customer analysis, recall might be more important if missing a positive case (e.g., identifying a loyal customer) is costly.

F1-Score: A balanced metric that is useful when you need to take both precision and recall into account.

5.2Evaluation Results:
Promotion as a feature has the best performing score as seen in Table 4 with the use of the algorithms. Total_Items has the lowest performing score among all the features.  This gives an insight that there is a correlation on how customers tend to purchase products if given a promotion.  The city feature depends on the location for the sale of the products and the purchasing power of customers. 


6.0 PHASE SIX: DEPLOYMENT

6.1 Deployment Plan:

Integrate the recommendation system into the retail website/app.
Use customer segments to tailor marketing campaigns.
Implement sales prediction model to optimize inventory management.

6.2 Monitoring and Maintenance:
Continuously monitor model performance and update models with new data.
Gather feedback from users to improve recommendation system.
Regularly retrain models to adapt to changing customer behaviour and market trends.

6.3 Documentation and Reporting: 
We used Tableau Software to design a dashboard for reporting track the progress of our business. 
6.3.1 Key Performance Indicators: (KPIs) Requirements 

1.Total Items:
Understand the overall landscape of items sold to assess the market's size and growth.

2.Total Cost:
Determine the cost of the products within per transaction giving a snippet how price of items is encompassed.

3.Total Sales:
Identify and analyse the total sales of in the dataset.
Calculate the percentage of total sales relative to the total number of items and the total cost.

4. Average Sales:
Identify and analyse the total number of sales in the dataset with the whole totality in the dataset.

6.3.2 Charts
1. Total Cost by Year:
Visualization: Line/ Area Chart

Description: This chart will illustrate the distribution of cost over the years, starting from 2020 to 2024, providing insights into the growth pattern and adoption trends. 
NB: We had to extract the year from the date column in the dataset

2. Total Sales by City:
Visualization: Map Chart

Description: This chart will showcase the geographical distribution of total sales across different states, allowing for the identification of regions with higher adoption rates.

3. Top 10 Total Sales by Product:
Visualization: Bar Chart

Description: Highlight the top 10 – 15 products based on the total number of sales, providing insights into the market dominance of specific brands.

4. Total Sales by Gender:
Visualization: Pie Chart or Donut Chart

Description: Illustrate the proportion of gender to provide incentives, aiding in understanding the impact of incentives on sales adoption.

6. Top Sales by Payment Method:
Visualization: Tree map

Description: Highlight the payment method based on the total number of sales, offering insights into consumer preferences and popular promotion in the market.
