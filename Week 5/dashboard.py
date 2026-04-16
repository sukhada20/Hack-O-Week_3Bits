import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="HVAC Optimization Dashboard", layout="wide")

st.title("HVAC Optimization in Labs")
st.write("Predict cooling needs based on occupancy and temperature")

# Load data
df = pd.read_csv("hvac_data.csv")

# Encode labels
le_lab = LabelEncoder()
df["lab_id_encoded"] = le_lab.fit_transform(df["lab_id"])

le_cool = LabelEncoder()
df["cooling_encoded"] = le_cool.fit_transform(df["cooling_needed"])

# Train model
X = df[["lab_id_encoded", "occupancy", "temperature"]]
y = df["cooling_encoded"]

model = DecisionTreeClassifier()
model.fit(X, y)

# =========================
# MODEL EVALUATION
# =========================
y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Input Parameters")

lab_input = st.sidebar.selectbox("Select Lab", df["lab_id"].unique())
occupancy_input = st.sidebar.slider("Occupancy", 0, 40, 15)
temp_input = st.sidebar.slider("Temperature (°C)", 20, 32, 26)

lab_encoded = le_lab.transform([lab_input])[0]
prediction = model.predict([[lab_encoded, occupancy_input, temp_input]])
cooling_result = le_cool.inverse_transform(prediction)[0]

# =========================
# PREDICTION OUTPUT
# =========================
st.subheader("Prediction Result")
st.success(f"Cooling Requirement: {cooling_result}")

# =========================
# GRAPH 1: BAR GRAPH (FIXED)
# =========================
st.subheader("Cooling Level Distribution")

cooling_counts = df["cooling_needed"].value_counts().reset_index()
cooling_counts.columns = ["Cooling Level", "Count"]

bar_fig = px.bar(
    cooling_counts,
    x="Cooling Level",
    y="Count",
    title="Count of Cooling Levels"
)

st.plotly_chart(bar_fig, use_container_width=True)

# =========================
# GRAPH 2: SCATTER PLOT
# =========================
st.subheader("Occupancy vs Temperature")

scatter_fig = px.scatter(
    df,
    x="occupancy",
    y="temperature",
    color="cooling_needed",
    title="Occupancy vs Temperature"
)

st.plotly_chart(scatter_fig, use_container_width=True)

# =========================
# HEATMAP
# =========================
st.subheader("Zone-wise Cooling Heatmap")

heatmap_df = df.groupby("lab_id")["cooling_encoded"].mean().reset_index()

fig = px.imshow(
    [heatmap_df["cooling_encoded"]],
    labels=dict(x="Lab", color="Cooling Level"),
    x=heatmap_df["lab_id"],
    y=["Average"],
    aspect="auto"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# MODEL METRICS
# =========================
st.subheader("Model Evaluation Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Precision", f"{precision*100:.2f}%")
col3.metric("Recall", f"{recall*100:.2f}%")
col4.metric("F1 Score", f"{f1*100:.2f}%")

# =========================
# CONFUSION MATRIX
# =========================
st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)

fig2, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig2)

# =========================
# DATASET VIEW
# =========================
with st.expander("View Dataset"):
    st.dataframe(df.head(50))