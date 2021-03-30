import pandas as pd     
import plotly           
import plotly.express as px
import plotly.io as pio
import collections
import plotly.graph_objects as go
import pickle
import webbrowser
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output , State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.graph_objects as go
try:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    def load_data():
        global pickle_model
        file=open('pickle_model.pkl','rb')
        pickle_model=pickle.load(file)
        global vocab
        file = open('feature.pkl', 'rb') 
        vocab = pickle.load(file)
        global scrape
        scrape=pd.read_csv('final.csv')
        scrape=scrape.sample(100)
    def check_review(reviewText):
        transformer = TfidfTransformer()
        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
        vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
        return pickle_model.predict(vectorised_review)
    def create_ui():
        main_layout=dbc.Container(
                [html.Div([
            html.H1(id='Main_title', children = "Sentiment Analysis On Product Reviews",style={'textAlign': 'center'}),
            html.Br(),
            html.Div(children='Pie Chart of almost 4.5 lakhs scrapped reviews',style={'textAlign': 'center','color': 'dark'}),
            dcc.Graph(id='pieChart',figure=pie_chart(),style={'textAlign': 'center'}),
            dcc.Dropdown(id='dropdown',options=[{'label': i, 'value': i} for i in scrape.reviews.unique()],placeholder='Select Scrapped Review...',style={'textAlign': 'left','width': '100%'}),
            html.Br(),html.Br(),
            dcc.Textarea(id='textarea_review',children='',placeholder = 'Enter the review here.....',style={'textAlign':'center','width': '100%', 'height': 200},),
            html.Br(),html.Br(),
            dbc.Button("Submit", id="btnshow",outline=True, className="mr-1"),
            html.Br(),html.Br(),
            dbc.Alert(id="show",is_open=False,duration=4000),
            html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
        ])],
        className = 'text-center'
        )
        return main_layout
    @app.callback(
        Output("textarea_review", "value"),
        [Input("dropdown", "value")],
    )
    def drop(val):
        if val!=None:
            return val
        else:
            return ""
    @app.callback(
        Output("show", "is_open"),
        Output("show", "children"),
        Output("show", "color"),
        [Input("btnshow", "n_clicks")],
        [State("show", "is_open"),
        State( 'textarea_review'  ,   'value'  )],
    )
    def toggle_alert(n, is_open,Textarea):
        if n:
            global res
            global clr
            response=check_review(Textarea)
            if (Textarea =='' ):
                res = 'Enter Review'
                clr="info"
            elif (response[0] == 0):
                res = 'Negative'
                clr="danger"
            elif (response[0] == 1 ):
                res = 'Positive'
                clr="success"
            else:
                res= 'Unknown'
                clr="warning"
            return not is_open,res,clr
        else:
            return is_open,'',''
    def open_browser():
        webbrowser.open_new('http://127.0.0.1:8050/')

    def pie_chart():
        df=pd.read_csv('scrapedReviews.csv')
        df=pd.DataFrame(df['sentiment'])
        cnt=collections.Counter(df['sentiment'])
        x=cnt[0]
        y=cnt[1]
        labels = ['Positive Reviews','Negative Reviews']
        values = [x,y]
        fig = go.Figure(data=[go.Pie(labels=labels, values=values,hole=.6)])
        return fig
    def main():
        load_data()
        app.layout=create_ui()
        open_browser()
        app.run_server()
    if __name__=='__main__':
        main()
except(Exception):
    print(Exception)