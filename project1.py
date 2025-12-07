import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
st.title("Ecommerce Customer Behavior Analysis  ML Web App")
uploaded_file = st.file_uploader("Upload ecommerce_customer_behavior_dataset.csv", type=["csv"])
if uploaded_file:
    customer = pd.read_csv(uploaded_file)
    st.subheader("Raw Dataset")
    st.write(customer.head())
    customer['Date'] = pd.to_datetime(customer['Date'])
    encoding = ['Gender','Payment_Method','City','Device_Type','Product_Category']
    le = LabelEncoder()
    for col in encoding:
        customer[col] = le.fit_transform(customer[col].astype(str))
    st.subheader("Encoded Dataset")
    st.write(customer.head())
    numeric_cols = customer.select_dtypes(include=['number']).columns
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(customer[numeric_cols].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)
    x = customer[['Age','Unit_Price','Quantity','Discount_Amount','Total_Amount',
                  'Session_Duration_Minutes','Pages_Viewed','Delivery_Time_Days',
                  'Customer_Rating','Gender','Payment_Method','City','Device_Type','Product_Category']]
    y = customer['Is_Returning_Customer']
    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=38)
    model_name = st.selectbox("Select Model", [
        "Logistic Regression",
        "Decision Tree (Entropy)",
        "Decision Tree (Gini)",
        "Random Forest (Entropy)",
        "Random Forest (Gini)",
        "KNN"
    ])
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree (Entropy)":
        model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    elif model_name == "Decision Tree (Gini)":
        model = DecisionTreeClassifier(criterion="gini", max_depth=3)
    elif model_name == "Random Forest (Entropy)":
        model = RandomForestClassifier(criterion="entropy")
    elif model_name == "Random Forest (Gini)":
        model = RandomForestClassifier(criterion="gini")
    else:
        model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    st.subheader("Model Results")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred))
