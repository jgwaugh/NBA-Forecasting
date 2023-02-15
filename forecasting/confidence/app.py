import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import shap
import streamlit as st

from forecasting.confidence import get_predictor_model

sns.set()


with open("residuals.pkl", "rb") as fp:
    stats = pickle.load(fp)

predictor = get_predictor_model()


stat_names = [k for k, v in stats.items()]


st.write(
    """
#  Confidence Interval Model Validation

Uses [SHAP](https://shap.readthedocs.io/en/latest/) plots to explain the marginal
relationships between features and standard error sizes. 
"""
)


stat = st.selectbox("Select a statistic to view marginal correlations", stat_names)

X = stats[stat]
X = X[["x", "t"]]

model = predictor.get_model(stat)


explainer = shap.TreeExplainer(model)
shap_values = explainer(X)


f1 = plt.figure()
plt.scatter(X.values[:, 1], shap_values.values[:, 1])
plt.xlabel("Time")
plt.ylabel("SHAP value for Time")


f2 = plt.figure()
plt.scatter(X.values[:, 0], shap_values.values[:, 0])
plt.xlabel(stat)
plt.ylabel(f"SHAP value for {stat}")


col1, col2 = st.columns(2)

with col1:
    st.pyplot(f1)

with col2:
    st.pyplot(f2)
