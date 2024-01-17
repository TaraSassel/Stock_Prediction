import streamlit as st 
import yfinance as yf

from prophet import Prophet 
from datetime import date 

from prophet.plot import plot_plotly
from plotly import graph_objs as go 

START = "2020-1-1"
TODAY = date.today().strftime("%Y-%m-%d")


st.title("Stock Prediction")
stock_list = ["GOOG", "AAPL", "AMZN", "TSLA", "MSFT", "NVDA"]
stock_list.sort()  # Sort in-place

stocks = tuple(stock_list)
selected_stock = st.selectbox("Select stock", stocks)

n_years = st.slider("Years of prediction", 0.5, 3.0, 1.5, 0.5)
period = int(n_years*365)

@st.cache_data # So it gets saved
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True) # Makes date first column 
    return data

data_load_state = st.text("Load data ...")
data = load_data(selected_stock)
data_load_state.text("Loading data completed")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'], name='stock_close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting 
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())

st.markdown("#### Forecast figure")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.markdown("#### Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
