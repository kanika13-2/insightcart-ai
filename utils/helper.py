import pandas as pd 

def load_data():
    df = pd.read_csv("data/walmart.csv")
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def total_revenue(df):
    return df['purchase_amount'].sum()

def revenue_by_category(df):
    return df.groupby('category')['purchase_amount'].sum()

def top_products(df):
    return df.groupby('product_name')['purchase_amount'].sum().sort_values(ascending=False).head(10)