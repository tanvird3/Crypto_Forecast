# importing required libraries
import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from datetime import datetime
import plotly.express as px
import plotly.graph_objs as go
import pandas_datareader as pdr
from prophet import Prophet

# initiate the app
app = dash.Dash()
server = app.server

# get the cryptocurrency list
crypcode = pd.read_csv("crypcode.csv")
cryp_value = crypcode["currency code"].tolist()
cryp_label = crypcode["currency name"].tolist()

# set date for importing data
todate = datetime.today().strftime("%Y-%m-%d")
firstdate = pd.to_datetime(todate, format="%Y-%m-%d") - pd.to_timedelta(
    365 , unit="d"
)

# read more about inline-block & flex
# https://www.geeksforgeeks.org/what-is-the-difference-between-inline-flex-and-inline-block-in-css/
# read more about padding and margin
# https://stackoverflow.com/questions/35620419/what-is-difference-between-css-margin-and-padding

# format the app
colors = {"background": "#222A2A", "text": "#479B55", "box": "#7F7F7F"}

app.layout = html.Div(
    children=[
        html.H1(
            "Crypto Forecast",
            style={
                "textAlign": "center",
                "paddingTop": "30px",
                "verticalAlign": "bottom",
            },
        ),
        #         html.H2(
        #             "Forecast Commonly Traded Cryptocurrencies with Facebook's Prophet Procedure",
        #             style={
        #                 "textAlign": "center",
        #                 "paddingTop": "5px",
        #                 "verticalAlign": "middle",
        #             },
        #         ),
        
        # the from crypto dropdown
        html.Div(
            [
                html.H3("Select Crypto", style={"paddingRight": "30px"}),
                dcc.Dropdown(
                    id="SelectCrypto",
                    options=[{"label": i, "value": i} for i in cryp_value],
                    value="BTC",
                    clearable=False,
                    style={
                        "fontsize": 24,
                        "width": 90,
                        "color": colors["text"],
                    },  # this style controls the text inside the dropdown
                ),
            ],
            style={  
                "display": "inline-block",
                "verticalAlign": "top",
                "paddingLeft": "20px",
                "paddingBottom": "20px",
            }, 		# this style controls the layout of the dropdown box
        ),
        # the submit button
        html.Div(
            [
                html.H3("", style={"paddingRight": "30px", "paddingBottom": "20px"}),
                html.Button(
                    id="submit-button",
                    n_clicks=0,
                    children="Forecast",
                    style={
                        "fontSize": 15,
                        "height": 30,
                        "verticalAlign": "middle",
                        "color": colors[
                            "text"
                        ],  
                    },	# this style controls the text inside the submit button
                )
            ],
            style={
                "display": "inline-block",
                "verticalAlign": "middle",
                "width": 90,
                # "height": 20,
                "paddingLeft": "7px",
                "paddingBottom": "20px",  
            },		# this style controls the entire submit button position etc.
        ),

        # the graphs
        dcc.Graph(id="forecast", style={"color": colors["text"]}),
    ],
    style={
        "backgroundColor": colors["background"],
        "color": colors["text"],
        "width": "100%",
        "height": "100%",  
    },			# this style controls the entire app
)

# app functions
@app.callback(
    Output(component_id="forecast", component_property="figure"),
    #         Output(component_id="Coef_Value", component_property="figure"),
    #         Output(component_id="Model_Evaluation", component_property="figure"),
    [Input("submit-button", "n_clicks")],
    [
        State("SelectCrypto", "value"),
        #         State("amount", "value"),
        #         State("isFlaggedSite", "value"),
        #         State("isUnrecognizedDevice", "value"),
        #         State("isOutsideLocation", "value"),
    ],
)

# start the function
def CryptoForecast(n_clicks, SelectCrypto):

   # get the data
    cryptodata = pdr.get_data_yahoo(
        [SelectCrypto + "-" + "USD"], start=firstdate, end=todate
    ).reset_index()
    cryptodata.columns = cryptodata.columns.get_level_values(0)

    cryp_pro = cryptodata[["Date", "Close"]]
    cryp_pro = cryp_pro.rename(columns={"Date": "ds", "Close": "y"})

    # Fit the Model
    model = Prophet()
    model.fit(cryp_pro)

    # make the prediction
    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)

    # prepare data for plotting 
    forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast = forecast.rename(columns={"ds": "Date"})
    plot_data = cryptodata.merge(forecast, on="Date", how="outer")
    plot_data["Volume"] = plot_data["Volume"].replace(np.nan, 0)
    plot_data = plot_data.iloc[-150:,:]

    # make the plot
    fig_forecast = px.scatter(
        plot_data,
        x="Date",
        y="Close",
        size="Volume",
        title=SelectCrypto,
        labels={"Date": "Date", "Close": "Closing Price"},
        template="plotly_dark",
    ).update_traces(mode="lines+markers", marker=dict(color="green", opacity=0.4))

    fig_forecast.add_trace(
        go.Scatter(
            x=plot_data["Date"],
            y=plot_data["yhat"],
            mode="lines",
            name="Predicted Price",
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=plot_data["Date"],
            y=plot_data["yhat_lower"],
            mode="lines",
            line=dict(dash="dot"),
            name="Lower Band",
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=plot_data["Date"],
            y=plot_data["yhat_upper"],
            mode="lines",
            line=dict(dash="dot"),
            name="Upper Band",
        )
    )

    return fig_forecast

# launch the app
if __name__ == "__main__":
    app.run_server(debug=False, threaded=False)