#Exploratory Data Analysis For Advance House Price 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

os.chdir(r"C:\Users\User\Downloads")
df = pd.read_csv("train (1).csv")
df.head(5)
df.info()

#Missing values 
feature_with_na = [feature for feature in df.columns if df[feature].isnull().sum() > 1]
feature_with_na

for feature in feature_with_na : 
    data = df.copy()
    data[feature] = np.where(data[feature].isnull() , 1 , 0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
    
#Numerical variables
numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
df[numerical_features].info()

#Extract from them the time or date numerical values 
time_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]
df[time_feature].head()

df.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")
plt.show()

#Discrete variables 
discrete_features = [feature for feature in numerical_features if len(df[feature].unique()) < 25 and feature not in time_feature+['Id']]
df[discrete_features].head(5)

for feature in discrete_features:
    data = df.copy()
    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

#Continuous Variable
continuous_features = [feature for feature in numerical_features if feature not in discrete_features+time_feature+['Id']]
df[continuous_features].head()

#log transform to make them appear in a normal distrbution 
for feature in continuous_features:
    data = df.copy()
    data[feature] = np.log(data[feature])
    data['SalePrice'] = np.log(data['SalePrice'])
    plt.scatter(data[feature], data['SalePrice'])
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

#Outliers
for feature in continuous_features:
    data=df.copy()
if 0 in data[feature].unique():
    pass
else:
    data[feature]=np.log(data[feature])
    data.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()
# Obesrvation : we have too many outliers 

#Categorical Variables
categorical_feature = [feature for feature in df.columns if df[feature].dtype == 'O']
df[categorical_feature].head()

for feature in categorical_feature:
   print(f"{feature} feature has {len(df[feature].unique())} unique values")


for feature in categorical_feature:
    data=df.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

# The Feature Engineering part 
#==============================

# Handle missing values
feature_with_na = [feature for feature in df.columns if df[feature].isnull().sum() > 1]
for feature in feature_with_na:
    data = df.copy()
    if df[feature].dtype != 'O':
        df[feature].fillna(df[feature].median(), inplace=True)
    else:
        df[feature].fillna(df[feature].mode()[0], inplace=True)

# One-hot encode categorical variables
categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Extract continuous numerical features
numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']
continuous_features = [feature for feature in numerical_features if feature not in ['Id']]

# Standardize continuous features
scaler = StandardScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])

df['Age'] = df['YrSold'] - df['YearBuilt']
df['Renovated'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

# Address outliers using the IQR method
for feature in continuous_features:
    data = df[feature]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    df[feature] = np.where(df[feature] < lower_limit, lower_limit, df[feature])
    df[feature] = np.where(df[feature] > upper_limit, upper_limit, df[feature])


X = df.drop(columns=['SalePrice', 'Id'])
y = df['SalePrice']

# Perform feature selection using Lasso Regression
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)
selected_features = X.columns[(lasso.coef_ != 0)]

# Create a subset of X with the selected features
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Now the data is ready for modeling
# ===================================


from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error , mean_absolute_error ,r2_score 

models = [('Linear Regression', LinearRegression(), {}),
          ('Random Forest Regressor', RandomForestRegressor(), 
           {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]}),
          ('Support Vector Machine', SVR(), 
           {'kernel': ['linear', 'sigmoid', 'rbf'], 'gamma': ['scale', 'auto'], 'C': [0.1, 1, 10]})
]

best_model = None
best_param = None
best_score = -np.inf

# Making cross validation to determine the best model and applying grid search to determine the best parameters

for name, model, model_param in models:
    grid_search = GridSearchCV(model, model_param, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"model: {name} --> best parameter: {grid_search.best_params_}")
    print(f"model: {name} --> best score: {grid_search.best_score_}")
    print("========================")
    
    if grid_search.best_score_ > best_score:
        best_model = model
        best_param = grid_search.best_params_
        best_score = grid_search.best_score_

print(f"best model: {best_model}")
print(f"best parameters: {best_param}")
print(f"best score: {best_score}")

# So now we can use Support Vector Machine becuase it the best model

svr_model = SVR(C=1 ,gamma='auto' , kernel='rbf')
svr_model.fit(X_train,y_train)
y_predicted = svr_model.predict(X_test)
y_predicted


#Evaluate our model 
#===================

# Scatter plot: Actual vs Predicted values
plt.scatter(y_test, y_predicted ,color='blue')
plt.plot([min(y_test),max(y_test)] , [min(y_test) ,max(y_test)] , color='red')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Actual values VS Predicted values')
plt.show()

#Residuals plot 
residuals = y_test - y_predicted 
plt.scatter(y_predicted , residuals , color='green')
plt.axhline(y=0 , color='red'  ,linestyle='--')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals VS Predicted values')
plt.show()

#Histogram for Residuals
plt.hist(residuals, bins=10, color='red', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()

#evaluation metrics values 
mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predicted)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R-squared (R²): {r2}')
