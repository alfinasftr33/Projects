# -*- coding: utf-8 -*-
"""
Created on Mon May  1 21:22:58 2023

@author: HP
"""
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
from nltk.collections import Counter
from wordcloud import WordCloud
import re
import numpy as np
import dash
import dash_html_components as html
import dash_core_components as dcc

df = pd.read_csv('https://raw.githubusercontent.com/alfina33/Sentiment-Analysis/main/svm_model.csv') 

app = dash.Dash(_name_)
server = app.server

app.layout = html.Div(
    children=[
        dcc.Input(id='text-input', type='text', placeholder='Enter text'),
        html.Button('Submit', id='submit-button', n_clicks=0),
        html.Div(id='result-div')
    ]
)

@app.callback(
    dash.dependencies.Output('result-div', 'children'),
    dash.dependencies.Input('submit-button', 'n_clicks'),
    dash.dependencies.State('text-input', 'value')
)
def update_output(n_clicks, input_value):
    if n_clicks > 0 and input_value is not None:
        analyzer = SentimentIntensityAnalyzer()
        sentiment = analyzer.polarity_scores(input_value)
        compound_score = sentiment['compound']
        if compound_score >= 0.05:
            sentiment_label = 'Positive'
        elif compound_score <= -0.05:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'

        vectorizer = TfidfVectorizer()
        x = vectorizer.fit_transform([input_value])
        random_forest_model = joblib.load('RandomForest.pkl')
        random_forest_sentiment = random_forest_model.predict(x)[0]

        if sentiment_label == 'Negative':
            cleaned_text = re.sub(r'[^a-zA-Z\s]', '', input_value.lower())

            # wordcloud
            wordcloud = WordCloud(width=800, height=400).generate(cleaned_text)
            wordcloud_image = wordcloud.to_image()

            # ngram
            tokens = nltk.word_tokenize(cleaned_text)
            n = 2
            ngram_freq = Counter(ngrams(tokens, n))
            labels, counts = zip(*ngram_freq.items())
            indices = np.arange(len(labels))

            # barplot
            plt.bar(indices, counts)
            plt.xticks(indices, labels, rotation='vertical')
            plt.tight_layout()
            plt.savefig('static/bar_plot.png')
            return html.Div([
                html.H4('Sentiment Analysis Result:'),
                html.P(f'Sentiment Label: {sentiment_label}'),
                html.P(f'Random Forest Sentiment: {random_forest_sentiment}'),
                html.Img(src='static/bar_plot.png'),
                html.Img(src=wordcloud_image)
            ])

        return html.Div([
            html.H4('Sentiment Analysis Result:'),
            html.P(f'Sentiment Label: {sentiment_label}'),
            html.P(f'Random Forest Sentiment: {random_forest_sentiment}')
        ])

    return html.Div()

if _name_ == '_main_':
    app.run_server(debug=True)


#----

# load data
df = pd.read_csv('https://raw.githubusercontent.com/alfina33/Sentiment-Analysis/main/svm_model.csv') 

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children='Data Science Salaries 2020-2023', style={
            'textAlign': 'center', 'color': '#CD5C5C'}),
    html.Div(children='''Data Source: https://www.kaggle.com/datasets/arnabchaki/data-science-salaries-2023''',
             style={'textAlign': 'center'}),
    dcc.Dropdown(df.job_title.unique(), 'Data Scientist',
                 id='dropdown-job_title', placeholder='Select a job title'),
    dcc.Dropdown(df.company_location.unique(), 'US',
                 id='dropdown-company_location', placeholder='Select a company location'),
    dcc.Graph(id='histogram'),
    dcc.Graph(id='heatmap'),
    dcc.Graph(id='bar-chart'),
    dcc.Graph(id='pie-chart'),
])

@callback(
    [Output('histogram', 'figure'),
     Output('heatmap', 'figure'),
     Output('bar-chart', 'figure'),
     Output('pie-chart', 'figure')],
    [Input('dropdown-job_title', 'value'),
     Input('dropdown-company_location', 'value')]
)

def update_graph(job_title_value, company_location_value):
    dff1 = df[(df['job_title'] == job_title_value)]
    dff2 = df[(df['job_title'] == job_title_value) & (df['company_location'] == company_location_value)]
    histogram = px.histogram(dff2, x='salary_in_usd', color='job_title', nbins=30, title='Salary Distribution Job Title')
    grup1 = dff1.groupby(['work_year', 'job_title', 'experience_level'])['salary_in_usd'].mean().reset_index()
    heatmap = px.imshow(grup1.pivot_table(index='experience_level', columns='job_title', values='salary_in_usd'),
                        title='Salary by Job Title and Experience Level')
    grup2 = dff1.groupby(['work_year', 'job_title', 'employment_type'])['salary_in_usd'].mean().reset_index()
    bar_chart = px.bar(grup2, x='employment_type', y='salary_in_usd', color='work_year',
                       title='Salary by employment type')
    pie_chart = px.pie(dff1, values='salary_in_usd', names='remote_ratio')
    
    return histogram, heatmap, bar_chart, pie_chart

if __name__ == '__main__':
    app.run_server(debug=True)