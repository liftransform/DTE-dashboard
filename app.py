import dash
from dash import html,dcc,Input,Output
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import plotly_express as px
import nltk
from string import punctuation

df = pd.read_excel('Dataset.xlsx')
colors  = ['#15616D','#BC5308', '#FFECD1', '#C5CAB8', '#FF7D00', '#8AA79F', '#FFB569', '#15616D', '#001524','#001524']
type_counts = df['Type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

fig = px.bar(type_counts, x='Count', y='Type', text='Count', orientation='h', 
             template='plotly_white', color_discrete_sequence=colors[3:])

fig.update_traces(textposition='inside')
fig.update_layout(xaxis_title='Count', yaxis_title='Type', title='Counts of Each Type')
fig.update_layout(title_text='The distribution of each study in the dataset', title_x=0.5, title_y=0.95)

y = df['Study Results'].value_counts().reset_index()
results  = df.groupby(['Study Results','Type']).size().reset_index().rename(columns = {0:'n'})
fig1 = px.bar(results,x ='Type',y ='n',color = 'Study Results' ,color_discrete_sequence=colors[:3]+colors[-1:], template='plotly_white')
fig1.update_layout(title_text='Does the study have results', title_x=0.5, title_y=0.95)

cond = df.groupby(['Conditions', 'Type']).size().reset_index().rename(columns={0: 'n'}).sort_values('n', ascending=False).head(11)
cond = cond.drop_duplicates(subset='Conditions')

fig2 = px.bar(cond, y='Conditions', x='n', text='n', title='Top 10 Conditions Under Study in Clinical Trials',color_discrete_sequence=colors,template = 'simple_white')
fig2.update_traces(textposition='inside', textfont_color='white')  # Add white text inside the bars
fig2.update_layout(title_x=0.5, title_y=0.95)

# Define stopwords
nltk.download('punkt')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english') + list(punctuation) + [str(i) for i in range(10)]

# Function to extract bigrams
def extract_bigrams(data, measure, study_type):
    # Extract the text
    text = " ".join([str(i) for i in data])

    # Extract the type
    tokens = nltk.word_tokenize(text.lower())

    # Remove the stopwords
    words = [i for i in tokens if i not in stopwords]

    # Create bigrams
    bigrams = list(nltk.bigrams(words))

    # Count bigrams
    bigram_freq = nltk.FreqDist(bigrams)

    # Get the top 10 bigrams
    top_10_bigrams = bigram_freq.most_common(10)
    dy = pd.DataFrame(top_10_bigrams, columns=['Bigram', 'Frequency'])
    dy['Type'] = study_type
    return dy

# Extract bigrams for all types
bgms = []
for study_type in df['Type'].unique():
    type_df = df[df['Type'] == study_type]
    primary_outcomes = type_df['Primary Outcome Measures']
    bgms.append(extract_bigrams(primary_outcomes, 'Primary Outcome Measures', study_type))

bigs = pd.concat(bgms)

# Modify the 'Bigram' column to be more readable
bigs['Bigram'] = [' '.join(i) for i in bigs['Bigram']]

# 'bigs' is your DataFrame containing the bigrams and frequencies
fig3 = px.bar(bigs, y='Bigram', x='Frequency', color='Type', facet_col='Type',
              title="Bigrams extracted from the Primary Outcome Measures of each study",
              color_discrete_sequence=colors, template='plotly_white', height=1000)
fig3.update_layout(title_x=0.5)

sponsors = df.groupby(['Sponsor', 'Type']).size().reset_index().rename(columns={0: 'n'}).sort_values('n', ascending=False)
top_5 = []
for i in sponsors['Type'].unique():
    dy = sponsors[sponsors['Type'] == i]
    top_5.append(dy.head(5))
all_sponsors = pd.concat(top_5)

# Assuming you have a color list named 'colors'
fig4 = px.bar(all_sponsors, x='Sponsor', y='n', color='Type', template='simple_white', color_discrete_sequence=colors,
             title="Top 5 sponsors for each study type", height=500)
fig4.update_layout(title_x=0.5)  # Center the title

collaborators = df.groupby(['Collaborators','Type']).size().reset_index().rename(columns = {0:'n'}).sort_values('n',ascending = False)
top_5 = []
for i in collaborators['Type'].unique():
    dy = collaborators[collaborators['Type']==i]
    top_5.append(dy.head(5))
all_collaborators = pd.concat(top_5)
fig5 = px.bar(all_collaborators,x = 'Collaborators',y = 'n',color = 'Type',template = 'simple_white',color_discrete_sequence=colors,title = "Top 5 collaborators for each study type",height =500)
fig5.update_layout(title_x=0.5)

def Bivariate(df, variable, selected_type):
    study = df[df['Type'] == selected_type]
    data = study.groupby([variable, 'Type']).size().reset_index().rename(columns={0: 'n'})
    fig = px.pie(data, names=variable, values='n', color_discrete_sequence=colors, template='plotly_white',
                 hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False)
    fig.update_layout(title_text=f'What is the {variable} distribution within the {selected_type} study?',
                      title_x=0.5, title_y=0.95)
    return fig


# box plot

# empty = []
# for i in df['Type'].unique():
#     new = df[df['Type'] == i]
#     Q1 = new['Enrollment'].quantile(0.25)
#     Q3 = new['Enrollment'].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_fence = Q1 - 1.5 * IQR
#     upper_fence = Q3 + 1.5 * IQR
#     df1 = new[(new['Enrollment'] >= lower_fence) & (new['Enrollment'] <= upper_fence)]
#     empty.append(df1)

# df2 = pd.concat(empty)


app=dash.Dash(__name__)

# Create the dropdown options from the unique values in the "Type" column
type_options = [{'label': type, 'value': type} for type in df['Type'].unique()]

app.layout = html.Div(
    style={'backgroundColor': '#C5CAB8', 'color': 'black'},
    children=[
       
       dcc.Dropdown(
            id='type-dropdown',
            options=type_options,
            value=type_options[0]['value'],
            style={
                'width': '99%',
                'height': '10%',
                'border': '1px solid black',
                'margin': 'auto',  # Center the dropdown horizontally
                'color': 'blue',   # Set text color to blue
                'background-color': 'lightblue',  # Set background color to light blue
                'font-family': 'Arial, sans-serif',  # Choose an appropriate font
                'font-size': '16px',  # Adjust the font size
                'border-radius': '5px'  # Add rounded corners for a visually appealing look
            }
        ),
             
     html.Div(dcc.Graph(id="type-count-bar-chart",figure=fig),style = {'width':"48%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),

     html.Div(dcc.Graph(id="001",figure=fig1),style = {'width':"48%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),

    html.Div(dcc.Graph(id="002",figure=fig2),style = {'width':"48%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),
    # Set the default selected value
    html.Div(dcc.Graph(id='bigrams-bar-chart', figure=fig3),style = {'width':"48%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),

    html.Div(dcc.Graph(id='sponsors-bar-chart',figure=fig4),style = {'width':"48%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},), 

    html.Div(dcc.Graph(id='collaborators-bar-chart',figure=fig5),style = {'width':"48%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},), 
                                              
    html.Div([
        dcc.Dropdown(id='variable-dropdown',options=[{'label': col, 'value': col} for col in df.columns],
        value='Sex'  # Set an initial value
    ), 
        dcc.Graph(id='bivariate-pie-chart')],style = {'width':"32%","height":"35%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),                                                                                                                               

    html.Div(dcc.Graph(id='boxplot-enrollment'),style = {'width':"32%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),

    html.Div(dcc.Graph(id='boxplot-months'),style = {'width':"32%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),

    html.Div(dcc.Graph(id='study-design-bar-chart'),style = {'width':"97%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),   

    html.Div(dcc.Graph(id='scatter-plot'),style = {'width':"98%","height":"40%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),   
    html.Div(dcc.Graph(id='geo'),style = {'width':"45%","height":"60%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),   
    html.Div(dcc.Graph(id='chro'),style = {'width':"45%","height":"60%","display":"inline-block",
                                               "border": "1px solid black","margin-right": "10px"},),  
                                                                                                                                            
])

@app.callback(
    Output('bigrams-bar-chart', 'figure'),
    Input('type-dropdown', 'value')
)
def update_bigrams_bar_chart(selected_type):
    filtered_bigrams = bigs[bigs['Type'] == selected_type]

    updated_fig = px.bar(filtered_bigrams, y='Bigram', x='Frequency', color='Type', facet_col='Type',
                         title="Bigrams extracted from the Primary Outcome Measures of each study",
                         color_discrete_sequence=colors, template='plotly_white')
    updated_fig.update_layout(title_x=0.5)

    return updated_fig

@app.callback(
    Output('sponsors-bar-chart', 'figure'),
    Input('type-dropdown', 'value')
)
def update_sponsors_bar_chart(selected_type):
    filtered_sponsors = all_sponsors[all_sponsors['Type'] == selected_type]

    updated_fig = px.bar(filtered_sponsors, y='Sponsor', x ='n', color='Type', template='simple_white',
                         color_discrete_sequence=colors, title="Top 5 sponsors for each study type", height=500)
    updated_fig.update_layout(title_x=0.5)  # Center the title

    return updated_fig

@app.callback(
    Output('collaborators-bar-chart', 'figure'),
    Input('type-dropdown', 'value')
)
def update_collaborators_bar_chart(selected_type):
    filtered_collaborators = all_collaborators[all_collaborators['Type'] == selected_type]

    updated_fig = px.bar(filtered_collaborators, y='Collaborators', x='n', color='Type', template='simple_white',
                         color_discrete_sequence=colors, title="Top 5 Collaborators for each study type", height=500)
    updated_fig.update_layout(title_x=0.5)  # Center the title

    return updated_fig

@app.callback(
    Output('bivariate-pie-chart', 'figure'),
    Input('type-dropdown', 'value'),
    Input('variable-dropdown', 'value',
    )
)
def update_bivariate_pie_chart(selected_type, selected_variable):
    df_copy = df.copy()  # Create a copy of the DataFrame to ensure proper alignment
    fig = Bivariate(df_copy, selected_variable, selected_type)
    return fig

@app.callback(
    Output('boxplot-enrollment', 'figure'),
    Input('type-dropdown', 'value')
)
def update_boxplot(selected_type):
    # df2 = df2[df2['Enrollment']<1500]
    filtered_df = df[df['Type'] == selected_type]
    filtered_df =  filtered_df[ filtered_df['Enrollment']<1500]
    fig = px.box(filtered_df, x='Type', y='Enrollment', points="all", color='Type',color_discrete_sequence=colors[4:],
                 title=f"Enrollment Distribution for {selected_type} studies",
                 template="simple_white")

    # Update the x-axis to add distribution
    fig.update_xaxes(showline=True, showgrid=False, title_text="Distribution")

    # Center the title
    fig.update_layout(title_x=0.5)

    # Annotate each boxplot with median value
    for i, box_data in enumerate(fig['data']):
        type_name = box_data['name']
        median = filtered_df['Enrollment'].median()
        fig.add_annotation(
            text=f'Median: {median:.2f}',
            x=i + 1,
            y=median,
            showarrow=False,
            font=dict(size=10)
        )

    return fig

@app.callback(
    Output('boxplot-months', 'figure'),
    Input('type-dropdown', 'value')
)
def update_boxplot1(selected_type):
    # df2 = df2[df2['Enrollment']<1500]
    dy = df.copy()
    # Calculate the difference in months between 'Completion Date' and 'Start Date'
    dy['Completion Date'] = pd.DatetimeIndex(dy['Completion Date'])
    dy['Start Date'] = pd.DatetimeIndex(dy['Start Date'])
    dy['Months'] = (dy['Completion Date'] - dy['Start Date']).dt.days / 30
    filtered_df = dy[dy['Months']<700]
    filtered_df = dy[dy['Type'] == selected_type]
    filtered_df =  filtered_df[ filtered_df['Months']<1500]
    fig = px.box(filtered_df, x='Type', y='Months', points="all", color='Type',color_discrete_sequence=colors[5:],
                 title=f"Duration from Study Start to Completion<br> in Months: {selected_type} studies",
                 template="simple_white")

    # Update the x-axis to add distribution
    fig.update_xaxes(showline=True, showgrid=False, title_text="Distribution")

    # Center the title
    fig.update_layout(title_x=0.5)

    # Annotate each boxplot with median value
    for i, box_data in enumerate(fig['data']):
        type_name = box_data['name']
        median = filtered_df['Enrollment'].median()
        fig.add_annotation(
            text=f'Median: {median:.2f}',
            x=i + 1,
            y=median,
            showarrow=False,
            font=dict(size=10)
        )

    return fig

@app.callback(
    Output('study-design-bar-chart', 'figure'),
    Input('type-dropdown', 'value')
)
def update_study_design_bar_chart(selected_type):
    y = df.groupby(['Study Design', 'Type']).size().reset_index().rename(columns={0: 'n'}).sort_values('n', ascending=False)
    filtered_data = y[y['Type'] == selected_type]
    new = []
    for i in filtered_data['Type'].unique():
        y1 = filtered_data[filtered_data['Type'] == i]
        new.append(y1.head(5))

    n = 10

    def cutter(text, n=100):
        new = " "
        for i in range(0, len(text), n):
            cut = text[i:i + n]
            new += cut + '<br>'
        return new

    y2 = pd.concat(new)
    y2['Study Design'] = y2['Study Design'].apply(cutter)
    fig = px.bar(y2, y='Study Design', x='n', color='Type', color_discrete_sequence=colors,
                 template='plotly_white', height=600)
    fig.update_layout(title_text=f'Most common study designs for {selected_type} studies',
                      title_x=0.5, title_y=0.95)
    return fig

@app.callback(
    Output('scatter-plot', 'figure'),
    Input('type-dropdown', 'value')
)
def update_scatter_plot(selected_type):
    y = df[['Start Date', 'Type']]
    y['Start Date'] = pd.DatetimeIndex(y['Start Date'])
    y1 = y[y['Type'] == selected_type]
    y1 = y1.groupby(['Start Date', 'Type']).size().reset_index().rename(columns={0: 'n'}).sort_values('n')

    fig = px.scatter(y1, x='Start Date', y='n', color='Type', template='plotly_white',color_discrete_sequence=colors[4:],
                     title=f"Scatter Plot of Study Start Dates Over Time: {selected_type} studies")

    return fig

@app.callback(
    Output('geo', 'figure'),
    Input('type-dropdown', 'value')
)
def update_geo_plot(selected_type):
    #-----------------------------------------------------------------------------------------------
    colors  = ['#15616D','#BC5308', '#FFECD1', '#C5CAB8', '#FF7D00', '#8AA79F', '#FFB569', '#15616D', '#001524','#001524']
    loc = df[['Type','Locations']]
    country = []
    for i in loc['Locations']:
        count = str(i).split(',')[-1]
        country.append(count)
    loc['Country'] = country
    loc1 = loc.dropna()
    y = loc1.groupby(['Type','Country']).size().reset_index().rename(columns = {0:'n'})
    dy = y[y['Type']==selected_type]
    i=selected_type
    fig8 = px.scatter_geo(
        dy,
        locations='Country',
        locationmode='country names',
        facet_col='Type',color = 'Type',
        size='n',color_discrete_sequence=colors[4:],
        title=f'Scatter Map of Studies by Type and Country:{selected_type} studies',
        template='simple_white',
        projection='orthographic',#"natural earth"# orthographic
        height=600,  # Increase the height (adjust to your preference)
        width=800,
    )

    # Customize the map layout if needed
    #fig.update_geos(showcoastlines=True, coastlinecolor=colors[0])

    fig8.update_geos(
    resolution=50,
    showcoastlines=True, coastlinecolor=colors[0],
    showland=True, landcolor=colors[2],
    showocean=True, oceancolor="#87CEEB",
    showlakes=True, lakecolor="#001524",
    showrivers=True, rivercolor="#001524"
    )

    # # Show the plot
    # fig8.update_layout(
    # font=dict(family="Arial Black", size=18, color="black"),
    # title_x=0.5,
    # title_y=0.95,
    # )
    fig8.update_traces(showlegend=False)
    return fig8

@app.callback(
    Output('chro', 'figure'),
    Input('type-dropdown', 'value')
)
def update_geo_plot(selected_type):
    #-----------------------------------------------------------------------------------------------
    colors  = ['#15616D','#BC5308', '#FFECD1', '#C5CAB8', '#FF7D00', '#8AA79F', '#FFB569', '#15616D', '#001524','#001524']
    loc = df[['Type','Locations']]
    country = []
    for i in loc['Locations']:
        count = str(i).split(',')[-1]
        country.append(count)
    loc['Country'] = country
    loc1 = loc.dropna()
    y = loc1.groupby(['Type','Country']).size().reset_index().rename(columns = {0:'n'})
    dy = y[y['Type']==selected_type]
    color_scale = px.colors.sequential.YlOrBr  # You can choose a different color scale
    min_scale = 0
    max_scale = 500
    i=selected_type
    fig8 =  px.choropleth(
        dy,
        locations='Country',
        locationmode='country names',
        color='n',
        hover_name='Country',
        title=f'Choropleth Map of Study Counts by Country:{i} Studies',
        color_continuous_scale=color_scale,
        color_continuous_midpoint=(max_scale + min_scale) / 2,  # Set the midpoint
        range_color=[min_scale, max_scale],  # Set the scale range
        template='simple_white',projection='natural earth',#"natural earth"# orthographic
        height=600,  # Increase the height (adjust to your preference)
        width=800,
    )

    # Customize the map layout if needed
    #fig.update_geos(showcoastlines=True, coastlinecolor=colors[0])

    fig8.update_geos(
    resolution=50,
    showcoastlines=True, coastlinecolor=colors[0],
    showland=True, landcolor=colors[2],
    showocean=True, oceancolor="#87CEEB",
    showlakes=True, lakecolor="#001524",
    showrivers=True, rivercolor="#001524"
    )

    # # Show the plot
    # fig8.update_layout(
    # font=dict(family="Arial Black", size=18, color="black"),
    # title_x=0.5,
    # title_y=0.95,
    # )

    return fig8


if __name__ == '__main__':
    app.run_server(debug=True)








