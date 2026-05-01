import streamlit as st
import pandas as pd
from utils.helper import load_data, revenue_by_category, top_products
from models.ann_model import train_ann
from models.forecast import forecast_sales

# ================= UI STYLING =================
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

h1 {
    text-align: center;
    color: #0f172a;
}

h2, h3 {
    color: #1e293b;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

section[data-testid="stSidebar"] {
    background-color: #1e293b;
}

section[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.stRadio > div {
    background-color: #334155;
    padding: 8px;
    border-radius: 8px;
}

button {
    border-radius: 10px !important;
    background-color: #3b82f6 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🛒 Retail AI Dashboard")

# ================= SIDEBAR =================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Chatbot", "AI Insights"])

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

# ================= DATA CLEANING =================
df.columns = df.columns.str.lower().str.replace(' ', '_')

# AUTO DETECT COLUMNS
amount_col = next((c for c in df.columns if any(x in c for x in ['amount','price','sales','revenue'])), None)
category_col = next((c for c in df.columns if 'category' in c or 'type' in c), None)
product_col = next((c for c in df.columns if 'product' in c or 'item' in c or 'name' in c), None)
date_col = next((c for c in df.columns if 'date' in c), None)

if amount_col is None:
    st.error("❌ No amount column found")
    st.stop()

df['purchase_amount'] = df[amount_col]

if category_col:
    df['category'] = df[category_col]

if product_col:
    df['product_name'] = df[product_col]

if date_col:
    df['purchase_date'] = df[date_col]

# ================= DASHBOARD =================
if page == "Dashboard":

    st.markdown("## 📊 Business Overview")
    st.markdown("---")

    st.markdown("### 🔑 Key Performance Indicators")

    col1, col2, col3 = st.columns(3)

    col1.metric("💰 Total Revenue", f"{df['purchase_amount'].sum():,.0f}")
    col2.metric("👥 Customers", df.shape[0])
    col3.metric("📊 Avg Spend", f"{df['purchase_amount'].mean():.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Revenue by Category")
        st.bar_chart(revenue_by_category(df))

    with col2:
        st.subheader("🏆 Top Products")
        data = top_products(df)
        if data is not None:
            st.bar_chart(data)
        else:
            st.warning("No product data")

# ================= CHATBOT =================
elif page == "Chatbot":

    st.markdown("## 💬 AI Data Assistant")

    query = st.text_input("Ask your data anything...")

    if query:
        q = query.lower()

        if "revenue" in q:
            st.success(f"Total Revenue: {df['purchase_amount'].sum():,.0f}")

        elif "customers" in q:
            st.success(f"Total Customers: {df.shape[0]}")

        elif "top product" in q:
            data = top_products(df)
            if data is not None:
                st.write(data.head(5))
            else:
                st.warning("No product data")

        else:
            st.info("Try: revenue, customers, top products")

# ================= AI INSIGHTS =================
elif page == "AI Insights":

    st.markdown("## 🤖 AI Insights & Predictions")

    tab1, tab2, tab3 = st.tabs(["🛒 Recommendation", "🤖 ANN", "📈 Forecast"])

    # -------- TAB 1 --------
    with tab1:
        st.subheader("Top Recommended Products")

        if 'product_name' in df.columns:
            st.write(top_products(df))
        else:
            st.warning("No product column")

    # -------- TAB 2 --------
    with tab2:
        st.markdown("### 🤖 Customer Intelligence")

        if 'age' in df.columns and 'purchase_amount' in df.columns:

            if st.button("🚀 Train AI Model", key="ann_btn"):
                model = train_ann(df)
                st.success("Model trained successfully!")

                sample = df[['age', 'purchase_amount']].iloc[0:1]
                pred = model.predict(sample)

                if pred[0][0] > 0.5:
                    st.success("High Value Customer 💎")
                else:
                    st.info("Normal Customer")

        else:
            st.warning("Need age + purchase_amount")

    # -------- TAB 3 --------
    with tab3:
        st.subheader("📈 Sales Forecast")

        if 'purchase_date' in df.columns and 'purchase_amount' in df.columns:

            if st.button("🚀 Run Forecast", key="forecast_btn"):
                result = forecast_sales(df)

                result['ds'] = pd.to_datetime(result['ds'])
                result['yhat_smooth'] = result['yhat'].rolling(10).mean()

                st.line_chart(result.set_index('ds')['yhat_smooth'])

        else:
            st.warning("Need date + amount")