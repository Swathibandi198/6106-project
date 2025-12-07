import pandas as pd

customer=pd.read_csv('ecommerce_customer_behavior_dataset.csv')

customer.info()

customer.isnull().sum()

customer.duplicated().sum()

customer['Date']=pd.to_datetime(customer['Date'])

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
encoding=['Gender','Payment_Method','City','Device_Type','Product_Category']
le=LabelEncoder()
for i in encoding:
    customer[i]=le.fit_transform(customer[i])
    print(customer)

sns.boxplot(data=customer)

x=['Date', 'Age', 'Unit_Price', 'Quantity', 'Discount_Amount', 'Total_Amount', 'Session_Duration_Minutes', 'Pages_Viewed','Delivery_Time_Days', 'Customer_Rating','Gender','Payment_Method','City','Device_Type','Product_Category']
for i in x:
    q1 = customer[i].quantile(0.25)
    q3 = customer[i].quantile(0.75)
    iqr = q3 - q1   
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = customer[(customer[i] < lower_bound) | (customer[i] > upper_bound)]
    percentage_outliers = (len(outliers) / len(customer)) * 100
    if percentage_outliers<=5:
        outliers_handled= customer[~((customer[i] < lower_bound) | (customer[i] > upper_bound))]
        print(f'Outliers handled for {i}, percentage of outliers: {percentage_outliers:.2f}%')
    print(f"Percentage of outliers in {i}: {percentage_outliers:.2f}%")

#EDA
import matplotlib.pyplot as plt

sns.barplot(customer['Payment_Method'].value_counts(),palette='pastel')
plt.show()

sns.barplot(customer['City'].value_counts(),palette='pastel')
plt.show()

sns.barplot(customer['Is_Returning_Customer'].value_counts(),palette='pastel')
plt.figure(figsize=(10,8))
plt.show()

sns.barplot(customer['Product_Category'].value_counts(),palette='pastel')
plt.show()

pair=customer.select_dtypes(include=['number'])
sns.pairplot(pair,palette='pastel')
plt.show()

corr=customer.corr(numeric_only=True)
plt.figure(figsize=(15,8))
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title("correlation heatmap")
plt.show()

x=customer[['Age','Unit_Price','Quantity','Discount_Amount','Total_Amount','Session_Duration_Minutes','Pages_Viewed','Delivery_Time_Days','Customer_Rating','Gender','Payment_Method','City','Device_Type','Product_Category']]
x

y=customer['Is_Returning_Customer']
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=38)

x_train.shape

x_test.shape

y_train.shape

y_test.shape

from sklearn.linear_model import LogisticRegression
m1=LogisticRegression()

m1.fit(x_train,y_train)

y_pred=m1.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score,classification_report
matrix=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
r1=recall_score(y_test,y_pred)
pre=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
classify=classification_report(y_test,y_pred)


acc

r1

pre

f1

print(classify)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
m1=DecisionTreeClassifier(criterion='entropy',max_depth=3)

m1.fit(x_train,y_train)

y_pred=m1.predict(x_test)
y_pred

tree.plot_tree(m1);

predict=m1.predict(x_test)
pd.Series(predict).value_counts()

predict.shape

accuracy=accuracy_score(y_test,y_pred)
accuracy

classify=classification_report(y_test,y_pred)
print(classify)

y_test.shape

m2=DecisionTreeClassifier(criterion='gini',max_depth=3)
m2


m2.fit(x_train,y_train)

y_pred=m2.predict(x_test)
y_pred

tree.plot_tree(m2)
plt.show()

predict=m2.predict(x_test)
pd.Series(predict).value_counts()

accuracy=accuracy_score(y_test,y_pred)
accuracy

classify=classification_report(y_test,y_pred)
print(classify)

from sklearn.ensemble import RandomForestClassifier
m3=RandomForestClassifier(criterion='entropy')
m3.fit(x_train,y_train)



y_pred=m3.predict(x_test)
y_pred

predict=m3.predict(x_test)
pd.Series(predict).value_counts()

accuracy=accuracy_score(y_test,y_pred)
accuracy

classify=classification_report(y_test,y_pred)
print(classify)

m4=RandomForestClassifier(criterion='gini')
m4.fit(x_train,y_train)

y_pred=m4.predict(x_test)
y_pred

predict=m4.predict(x_test)
pd.Series(predict).value_counts()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
accuracy

new=[[25,1,2,3,150.0,2,10.0,290.0,0,1,15,5,2,4]]
prediction=knn.predict(new)
print([prediction][0])

