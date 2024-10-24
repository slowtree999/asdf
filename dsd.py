import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


st.title("Iris Species Predictor")
st.write("""
This app predicts the Iris flower species based on the input parameters.
You can adjust the parameters using the sliders and see the predicted species.
Additionally, it provides various visualization of the dataset.
         """)

iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]


st.sidebar.header("Input Parameters")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length (cm)",
                                     float(df['sepal length (cm)'].min()),
                                     float(df['sepal length (cm)'].max()),
                                     float(df['sepal length (cm)'].mean())
                                     )
    sepal_width = st.sidebar.slider("Sepal width (cm)",
                                     float(df['sepal width (cm)'].min()),
                                     float(df['sepal width (cm)'].max()),
                                     float(df['sepal width (cm)'].mean())
                                     )
    petal_length = st.sidebar.slider("Petal length (cm)",
                                     float(df['petal length (cm)'].min()),
                                     float(df['petal length (cm)'].max()),
                                     float(df['petal length (cm)'].mean())
                                     )
    petal_width = st.sidebar.slider("Sepal length (cm)",
                                     float(df['petal width (cm)'].min()),
                                     float(df['petal width (cm)'].max()),
                                     float(df['petal width (cm)'].mean())
                                     )
    data = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)': petal_width
            }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()



st.subheader("User Input Parameters")
st.write(input_df)


model = RandomForestClassifier()
model.fit(X, y)


prediction = model.predict(input_df.to_numpy())
print(prediction, iris.target_names)

prediction_prob = model.predict_proba(input_df.to_numpy())

st.subheader("Prediction")
st.write(iris.target_names[prediction]) 
st.subheader("Prediction Probability")
st.write(prediction_prob)


st.subheader("Feature Importance")
importance = model.feature_importances_
print(importance)
indices = np.argsort(importance)
#print(indies)
plt.figure(figsize = (10, 4))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importance[indices])
plt.xticks(range(X.shape[1]), [iris.feature_names[i] for i in indices])

plt.xlim([-1, X.shape[1]])
st.pyplot(plt)


st.subheader("Histogram of Features")
fig, axes = plt.subplots(2, 2, figsize = (12, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    sns.histplot(df[iris.feature_names[i]], kde=True, ax=ax)
    ax.set_title(iris.feature_names[i])
plt.tight_layout()
st.pyplot(fig)


plt.figure(figsize = (10, 8))
st.subheader("Correlation Matrix")
numerical_df = df.drop("species", axis = 1)
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", fmt = '.2f', linewidths = 0.5)
plt.tight_layout()
st.pyplot(plt)


st.subheader('pairplot')
fig = sns.pairplot(df, hue = "species")
plt.tight_layout()
st.pyplot(fig)