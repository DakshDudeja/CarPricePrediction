import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#analysing the data type
dataset = pd.read_csv('CarPrice_Assignment.csv')
dataset.describe
dataset.shape
num=dataset.select_dtypes({'int64','float64'}).columns
num
dataset.select_dtypes({'int64','float64'}).shape
cat=dataset.select_dtypes({'object'}).columns
cat
#must yeh sab bata deta hai
dataset.info()

#correlation matrix
corrmat= dataset.corr()
plt.subplots(figsize=(10,10))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, center=0 ,cmap='Accent')

#check the negative relations in order to remove features and improve accuracy 
sns.heatmap(corrmat[(corrmat<-0.5)])
k=10
most_correlated=corrmat.nlargest(k, 'price')['price'].index
print(most_correlated)

#checking features with price to see any anomly or unusal change
sns.lmplot(x='enginesize', y='price',data=dataset)
sns.lmplot(x='curbweight', y='price', data=dataset)
sns.lmplot(x='horsepower', y='price', data=dataset)
sns.lmplot(x='carwidth', y='price', data=dataset)

#framing questions and answering them by analsing the graphs
#example why more price but less horsepower


#now dividing the company name and model name from the dataset
dataset['CarName']
CompanyName = dataset['CarName'].apply(lambda x : x.split(' ')[0])
dataset.insert(3,"CompanyName",CompanyName)
dataset.head()


#we will see if the data has unique names or not
dataset.CompanyName.unique()

#we will replace the same names with a common name or will correct the missspelled words 
def replace_name(a,b):
    dataset.CompanyName.replace(a,b,inplace=True)
replace_name('maxda','mazda')
replace_name('porcshce','porsche')
replace_name('toyouta','toyota')
replace_name('vokswagen','volkswagen')
replace_name('vw','volkswagen')

#analyze more features and check their frequences  
sns.countplot(dataset['fueltype'])
sns.countplot(dataset['carbody'])
sns.countplot(dataset['symboling'])
sns.boxplot(x=dataset['symboling'],y=dataset['price'])

#taking out fueleconomy using city and highway mpg by to make it short and crisp
dataset['fueleconomy'] = (0.55 * dataset['citympg']) + (0.45 * dataset['highwaympg'])
temp=dataset.copy()
table=temp.groupby(['CompanyName'])['price'].mean()
temp=temp.merge(table.reset_index(),how = 'left',on='CompanyName')
bins=[0,10000,20000,40000]
dataset_bin=['Budget','Medium','Highend']
dataset['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=dataset_bin)
dataset.head()

#dummy encoding
dataset1 = dataset[['price', 'fueltype', 'aspiration','carbody', 'drivewheel','wheelbase',
                  'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower', 
                    'fueleconomy', 'carlength','carwidth', 'carsrange']]
dataset1 = pd.get_dummies(data=dataset1,columns=['fueltype','aspiration','carbody','drivewheel','enginetype','cylindernumber','carsrange'],drop_first=True)

#dividing the data into train and test set
from sklearn.model_selection import train_test_split
dataset1_train,dataset1_test = train_test_split(dataset1,train_size=0.7,test_size=0.3, random_state=100)

#scaling the data in a range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
dataset1_train.loc[:,num_vars] = scaler.fit_transform(dataset1_train[num_vars])
dataset1_train.head()

#dividing the data into x and y and pop is used to remove out the mention element and return it 
y_train=dataset1_train.pop('price')
X_train=dataset1_train

#Colinearity is the state where two variables are highly correlated and contain similiar information about the variance within a given dataset.
#The Variance Inflation Factor (VIF) is a measure of colinearity among predictor variables within a multiple regression. 
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from statsmodels.stats.outliers_influence import variance_inflation_factor

regressor=LinearRegression()
regressor.fit(X_train,y_train)
rfe = RFE(regressor, 10)
rfe = rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

X_train_rfe = X_train[X_train.columns[rfe.support_]]
X_train_rfe.head()


def build_model(X,y):
    X = sm.add_constant(X) #Adding the constant
    regressor= sm.OLS(y,X).fit() # fitting the model
    print(regressor.summary()) # model summary
    return X

X_train_new = build_model(X_train_rfe,y_train)

X_train_new = X_train_rfe.drop(["cylindernumber_twelve"], axis = 1)

X_train_new = build_model(X_train_new,y_train)

X_train_new = X_train_new.drop(["fueleconomy"], axis = 1)

X_train_new = build_model(X_train_new,y_train)



#checking VIF
def checkVIF(X):
    vif = pd.DataFrame()
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return(vif)
    
    
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["curbweight"], axis = 1)

X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)




X_train_new = X_train_new.drop(["carbody_sedan"], axis = 1)


X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)

X_train_new = X_train_new.drop(["carbody_wagon"], axis = 1)

#ab sab kuch sahi lag raha hai
regressor = sm.OLS(y_train,X_train_new).fit()
y_train_price = regressor.predict(X_train_new)


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 

#predicition test set 
num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']
dataset1_test.loc[:,num_vars] = scaler.fit_transform(dataset1_test[num_vars])
y_test=dataset1_test.pop('price')
X_test=dataset1_test

X_train_new = X_train_new.drop('const',axis=1)
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = regressor.predict(X_test_new)

#EVALUATION OF THE MODEL
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)   



