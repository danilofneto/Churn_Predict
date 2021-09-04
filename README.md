# Churn predict

## Churn predict

#### This project was made by Danilo Felipe Neto.

# 1. Business Problem.

TopBank is a bank company that operates primarily in Europe. TopBank's main product is a bank account where the custumer doesn't have any cost in the first 12 months.

However, the client needs to rehire the bank for the upcoming 12 months after the free trial expires or everytime this new 12 months hired also expires.

Recently, the Analytics team noticed that the churn rate of clients is steadily increasing.

Worried about this new growing rate of churn clients, the Data Science team has the challenge to create an action plan to reduce the churn rate of clients.

# 2. The Challenge.
As a Data Science Consultant, you need to create an action plan to decrease the number of churn customers and show the financial return on your solution. At the end of your consultancy, you need to deliver to the TopBank CEO a model in production, which will receive a customer base via API and return that same base with an extra column with the probability of each customer going into churn. In addition, you will need to provide a report reporting your model's performance and the financial impact of your solution. Questions that the CEO and the Analytics team would like to see in their report:

* What is Top Bank's current Churn rate? How does it vary monthly?

* What is the performance of the model in classifying customers as churns?

* What is the expected return, in terms of revenue, if the company uses its model to avoid churn from customers?

A possible action to prevent the customer from churning is to offer a discount coupon, or some other financial incentive for him to renew his contract for another 12 months.

* Which customers would you give the financial incentive to and what would that amount be, in order to maximize ROI (Return on Investment). Recalling that the sum of incentives for each client cannot exceed EUR 10.000,00.

# 3. Solution Strategy

My strategy to solve this challenge was:

**Step 01. Data Description:**

The initial DataSet has 10000 rows and 14 columns. The features are the following:

* **RowNumber**: Row Number
* **CustomerID**: Customer Identification has no effect on customer leaving the bank.
* **Surname:** Customer's surname
* **CreditScore:** The customer's Credit score for the consumer market.
* **Geography:** The country where the customer live.
* **Gender:** Customer's gender
* **Age:** Customer's age.
* **Tenure:** Number of years the customer has remained with the active account.
* **Balance:** Monetary value the customer has in their bank account.
* **NumOfProducts:** The number of products purchased by the customer at the bank.
* **HasCrCard:** Indicates whether the customer has a credit card.
* **IsActiveMember:** Indicates whether the customer made at least one bank account transaction within 12 months.
* **EstimateSalary:** Estimate of the client's salary.
* **Exited:** whether or not the customer left the bank. (0=No,1=Yes) **Target variable.**


**Step 02. Feature Engineering:** New features like were created to make possible a more thorough analysis.

**Step 03. Data Filtering:** In this step the entries containing no information or containing information which does not match the scope of the project were filtered out.

**Step 04. Exploratory Data Analysis:** univariate, bivariate and multivariate data analysis, obtaining statistical properties of each of them.

**Step 05. Data Preparation:** Prepare the data so that machine learning models can learn specific behavior. 

**Step 06. Feature Selection:** The most statistically relevant features were selected using the Boruta package. The Boruta algorithm did not select the Geography variable, but in my opinion it is extremely important and was added in the selected features.

**Step 07. Machine Learning Modelling:** Some machine learning models were trained. The one that presented best results after cross-validation went through a further stage of hyperparameter fine tunning to optimize the model's generalizability.

**Step 08. Hyperparameter Fine Tunning:** Hyperparameter fine tunning to optimize the model's generalizability.

**Step 09. Convert Model Performance to Business Values:** The models performance is converted into business values.

**Step 10. Deploy Modelo to Production:** The model is deployed on a cloud environment to make possible that other stakeholders and services access its results. The model was incorporated into a google spreadsheet where it predicts the percentage of customers going into churn.
![image](references\google-sheets.PNG)


# 4. Top 3 Data Insights

**Hypothesis 01:** Younger customers tend to leave the bank easier.

**False:** Customers between 40 and 60 are more likely to churn.
![image](references\H1.png)

**Hypothesis 02:** Customers with fewer products tend to leave the bank easier.

**True:** People with 3 and 4 products are less likely to churn.
![image](references\H2.png)

**Hypothesis 03:** Customers without credit cards are more likely to churn.

**False.:** Customers with credit card are more likely to churn.
![image](references\H3.png)

# 5. Machine Learning Model Applied
The following machine learning models were trained:
* Logistic Regression
* Suport Vector Machine(SVM)
* XGBoost Classifier
* Random Forest

# 6. Machine Learning Modelo Performance
The chosen model was **XGBoost Classifier** because it have the better recall score(TP/TP+FN).
![image](references\model-performance.PNG)
![image](references\model-performance-ROC.png)


# 7. Business Results
After a cupom discount simulation ($100, $50, $25) based on the mean scenarios of the simulation, our best ROI(Return on Investiment) was applying discounts from $25.

* ROI $25: 26583%
* ROI $50: 21199%
* ROI $100: 13951%

# 8. Next Steps to Improve
In the next project cycle I will test new machine learning models and improve XGBoost fine tuning. Embed the model in a data visualization tool (Power BI, Tableau, Google Data Studio)



