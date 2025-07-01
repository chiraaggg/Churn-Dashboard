import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit_authenticator as stauth

# ✅ Load user credentials from TOML secrets
credentials = {
    "usernames": {
        user: {
            "email": st.secrets[f"credentials.usernames.{user}"].email,
            "name": st.secrets[f"credentials.usernames.{user}"].name,
            "password": st.secrets[f"credentials.usernames.{user}"].password,
        }
        for user in st.secrets["credentials"]["usernames"]
    }
}

# ✅ Setup Streamlit Authenticator
authenticator = stauth.Authenticate(
    credentials,
    "origin_churn_dashboard", "abcdef", cookie_expiry_days=1
)

# ✅ Login
name, auth_status, username = authenticator.login("🔐 Login", "main")

if auth_status is False:
    st.error("❌ Incorrect username or password")
elif auth_status is None:
    st.warning("🔐 Please enter your credentials")
elif auth_status:
    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"✅ Logged in as {name}")


st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide")

# 🚀 Title
st.title("📉 Customer Churn Prediction Dashboard - Origin ")

# 🔹 Load model and features
model, feature_names = joblib.load("lightgbm_churn_model.pkl")

# 🔹 Load CSV
uploaded_file = st.file_uploader("📁 Upload churn dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

# 🔹 Save user_id for output
user_ids = df['user_id'] if 'user_id' in df.columns else None

# 🔹 Drop unnecessary columns
df_model = df.drop(columns=['user_id', 'latitude', 'longitude'], errors='ignore')

# 🔹 One-hot encode if needed (optional safeguard)
if 'payment_method' in df_model.columns:
    dummies = pd.get_dummies(df_model['payment_method'], prefix='payment_method')
    df_model = pd.concat([df_model.drop(columns=['payment_method']), dummies], axis=1)

# 🔹 Ensure model compatibility
for col in feature_names:
    if col not in df_model.columns:
        df_model[col] = 0
df_model = df_model[feature_names]

# 🔹 Predict
churn_proba = model.predict_proba(df_model)[:, 1]

# 🔹 Apply threshold
threshold = st.slider("🔧 Churn Threshold", 0.0, 1.0, 0.4, 0.01)
predicted_churn = (churn_proba > threshold).astype(int)

# 🔹 Append results
df['churn_probability'] = churn_proba
df['predicted_churn'] = predicted_churn

# 🔹 KPIs
col1, col2, col3 = st.columns(3)
with col1:
    churn_rate = df['predicted_churn'].mean() * 100
    st.metric("📉 Churn Rate", f"{churn_rate:.2f}%")
with col2:
    if 'aov' in df.columns:
        st.metric("💰 Avg Order Value", f"{df['aov'].mean():.2f}")
with col3:
    if 'used_coupon_percent' in df.columns:
        st.metric("🎟️ Avg Coupon Usage %", f"{df['used_coupon_percent'].mean():.1f}%")

import plotly.express as px

# 🔹 Charts
st.subheader("📊 Visual Insights")

# 1. Churn probability distribution
fig1 = px.histogram(
    df,
    x="churn_probability",
    nbins=20,
    title="Distribution of Churn Probability",
    color_discrete_sequence=["#636EFA"],
    width=600,
    height=500,
    
)
fig1.update_layout(xaxis_title="Churn Probability", yaxis_title="Count")
st.plotly_chart(fig1, use_container_width=True)

# 2. Churn prediction count
fig2 = px.histogram(
    df,
    x="predicted_churn",
    title="Churn Prediction Distribution",
    color="predicted_churn",
    color_discrete_map={0: "#00CC96", 1: "#EF553B"},
    category_orders={"predicted_churn": [0, 1]},
    text_auto=True
)
fig2.update_layout(
    xaxis_title="Churn Prediction",
    yaxis_title="User Count",
    xaxis=dict(tickvals=[0, 1], ticktext=["Not Churned", "Churned"]),
    width=600,
    height=500,
    bargap=0.2,
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Boxplots for features by churn status
for feature in ['aov', 'used_coupon_percent', 'order_frequency_per_day']:
    if feature in df.columns:
        st.markdown(f"#### {feature} by Predicted Churn")
        fig_box = px.box(
            df,
            x='predicted_churn',
            y=feature,
            color='predicted_churn',
            points="all",
            color_discrete_map={0: "#00CC96", 1: "#EF553B"},
            category_orders={"predicted_churn": [0, 1]},
            labels={'predicted_churn': 'Churn Status'}
        )
        fig_box.update_layout(
            xaxis=dict(tickvals=[0, 1], ticktext=["Not Churned", "Churned"]),
            yaxis_title=feature,
        )
        st.plotly_chart(fig_box, use_container_width=True)

# 🔹 Full data table
st.subheader("📋 Churn Predictions Table")
df1 = df[['user_id', 'churn_probability', 'predicted_churn']]
df1['churn_probability'] = (df1['churn_probability'] * 100).round(2)
df1['risk_level'] = df1['churn_probability'].apply(lambda x: '⚠️ High' if x > threshold*100 else '✅ Low')
# Filter high-risk users only
high_risk_df = df[df['churn_probability'] > threshold][['user_id', 'churn_probability', 'predicted_churn']]

st.subheader("📋 High-Risk Users")
st.dataframe(high_risk_df)

st.download_button("⬇️ Download High-Risk Users CSV", high_risk_df.to_csv(index=False), "high_risk_users.csv")


