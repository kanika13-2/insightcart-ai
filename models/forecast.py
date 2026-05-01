import pandas as pd
from prophet import Prophet

def forecast_sales(df):
    df2 = df.groupby('purchase_date')['purchase_amount'].sum().reset_index()
    df2.columns = ['ds', 'y']

    df2['ds'] = pd.to_datetime(df2['ds'])

    model = Prophet()
    model.fit(df2)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    return forecast