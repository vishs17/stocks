import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


st.title('Stock Market Trends Predictor')


stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock1 = st.selectbox('Select stock for prediction', stocks)


n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data1 = load_data(selected_stock1)
data_load_state.text('Loading data... done!')


st.subheader(f'Raw data for {selected_stock1}')
st.write(data1.tail())


def plot_raw_data(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text=title, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

st.subheader(f'Raw data plot for {selected_stock1}')
plot_raw_data(data1, f'Time Series data for {selected_stock1} with Rangeslider')


def forecast_and_plot(data, period, stock_name):
   
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    

    m = Prophet()
    m.fit(df_train)
    

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    
  
    st.subheader(f'Forecast data for {stock_name}')
    st.write(forecast.tail())
    
    st.write(f'Forecast plot for {stock_name} for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    
   
    st.write(f"Forecast components for {stock_name}")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
    
    return forecast, m


forecast1, model1 = forecast_and_plot(data1, period, selected_stock1)


compare = st.checkbox('Compare with another stock')

if compare:
   
    selected_stock2 = st.selectbox('Select second stock for prediction', stocks)
    
   
    data_load_state = st.text('Loading data...')
    data2 = load_data(selected_stock2)
    data_load_state.text('Loading data... done!')
    
    
    st.subheader(f'Raw data for {selected_stock2}')
    st.write(data2.tail())
    
    st.subheader(f'Raw data plot for {selected_stock2}')
    plot_raw_data(data2, f'Time Series data for {selected_stock2} with Rangeslider')
    
   
    forecast2, model2 = forecast_and_plot(data2, period, selected_stock2)
    
   
    st.subheader('Investment Recommendation')
    
    max_high1 = forecast1['yhat'].max()
    max_high2 = forecast2['yhat'].max()
    
    if max_high1 > max_high2:
        recommendation = f"{selected_stock1} is expected to have a greater and more profitable stock high."
    else:
        recommendation = f"{selected_stock2} is expected to have a greater and more profitable stock high."
    
    st.write(recommendation)
    
    st.write(f'Predicted highest value for {selected_stock1}: {max_high1}')
    st.write(f'Predicted highest value for {selected_stock2}: {max_high2}')

   
    st.subheader('Combined Forecast Plot')
    
    def plot_combined_forecast(forecast1, forecast2, stock_name1, stock_name2):
        fig = go.Figure()
        

        fig.add_trace(go.Scatter(
            x=forecast1['ds'], y=forecast1['yhat'], name=f'{stock_name1} Forecast',
            line=dict(color='blue', width=2)
        ))
        
        
        fig.add_trace(go.Scatter(
            x=forecast2['ds'], y=forecast2['yhat'], name=f'{stock_name2} Forecast',
            line=dict(color='red', width=2)
        ))
        
        fig.layout.update(title_text=f'Combined Forecast for {stock_name1} and {stock_name2}',
                          xaxis_title='Date', yaxis_title='Stock Price', xaxis_rangeslider_visible=True)
        
        st.plotly_chart(fig)
    
    plot_combined_forecast(forecast1, forecast2, selected_stock1, selected_stock2)
