---
title: "Analysis of 2019 Philippine Elections"
excerpt: "Short description of portfolio item number 1<br/><img src='/images/500x300.png'>"
collection: portfolio
---

```python
import pandas as pd
import numpy as np
import seaborn as sns
import os
import glob
import geopandas
import re
import csv
import operator

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
import plotly.offline as pyoff

import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)
import matplotlib.pyplot as plt
%matplotlib inline
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
from IPython.display import HTML

HTML('''<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>''')
```




<script>
  function code_toggle() {
    if (code_shown){
      $('div.input').hide('500');
      $('#toggleButton').val('Show Code')
    } else {
      $('div.input').show('500');
      $('#toggleButton').val('Hide Code')
    }
    code_shown = !code_shown
  }

  $( document ).ready(function(){
    code_shown=false;
    $('div.input').hide()
  });
</script>
<form action="javascript:code_toggle()"><input type="submit" id="toggleButton" value="Show Code"></form>



# <center> A look into the Influence of Duterte on the 2019 Elections <br>and Voter Behavior</center>

## 2016 Presidential Elections

In the 2016 Presidential Elections, Rody Duterte ran on a populist platform, espousing and glorifying the story of the "common man" in politics, and in doing so became a dark horse that ultimately won in the election. In the intervening 3 years of his presidency, there have been many controversies that have sprung up; from the more mundane such as his various comments on various topics, to more serious controversies such as his stance on China and his war on drugs and its effects on the country.


Facing a lot of backlash from the opposition and media, Duterte seems to remain strongly positioned for power in the present and the future. From his roots, as a hands-on Mayor of Davao, to his presidency, he has continually espoused the fact that he is just a "common guy" who happens to be in a position of power. His populist ideas and platform brought him to the very top of the Philippine government and he has continued on with this stance through his presidency. Throughout the different controversies he's faced, how has this populist notion translated to support for him and his allies?

Let's take a look at the aforementioned 2016 presidential win by Duterte, broken down by the percentage of total votes per province he received:


```python
pres_16 = pd.read_csv('2016 Election Results/final_pres_geo_percent.csv')
ph1 = geopandas.GeoDataFrame.from_file("./Geopandas Shapes/gadm36_PHL_1.shp")
df_geo = ph1.merge(pres_16, left_on='NAME_1', right_on='geopanda_match')
```


```python
colors = 5
cmap = 'Reds'
figsize = (15, 15)
ax = df_geo.plot(column='DUTERTE, RODY (PDPLBN)', cmap=cmap, figsize=figsize, 
                          scheme='equal_interval', k=colors, legend=True, legend_kwds={'loc':2})
ax.set_title('% of Total Votes per Province - Duterte 2016');
```


![png](DMW%20Blog%20%231_files/DMW%20Blog%20%231_5_0.png)


In the choropleth above, we see that most of the votes in 2016 for Duterte came from Mindanao, which was his home region. In some cases, he won provinces overwhelmingly with over 90% of the votes in that province. What was it about his message of being the "common man" and leveraging on his populist approach helped him win?

Aside from being his home state, Mindanao also has one of the highest incidences of poverty in the country. Let's take a look at data from the latest 2015 Census for the top 10 provinces for which Duterte won majority of the votes and their poverty incidence:


```python
# load census data with poverty incidence
census_poverty = pd.read_excel('PSA Census Poverty Incidence.xlsx', skiprows=2, usecols=['Unnamed: 0', 'Unnamed: 10'])
census_poverty.dropna(how='any', inplace=True)
census_poverty = census_poverty[census_poverty['Unnamed: 0'].str.startswith('..')]
census_poverty.loc[104] = ['NCR', 3.0]
census_poverty = census_poverty.iloc[4:,:]
cleanup_col = [re.findall('\.?\.?([\w\s\.]*)', i)[0] for i in census_poverty['Unnamed: 0'].to_list()]
for i in range(len(cleanup_col)):
    if cleanup_col[i][-2:] == ' b' or cleanup_col[i][-2:] == ' c':
        cleanup_col[i] = cleanup_col[i][:-2]
    else:
        pass
census_poverty['Unnamed: 0'] = cleanup_col
census_poverty.columns = ['province', 'poverty_incidence']
census_poverty.sort_values(by='province', inplace=True)
keys = []
for i in cleanup_col:
    if i.lower() in ph1.NAME_1.str.lower().to_list():
        pass
    else:
        keys.append(i)
values = ['Mountain Province', 'Samar', 'Isabela', 'Sarangani', 
         'North Cotabato', 'Tawi-Tawi', 'Metropolitan Manila']
conv_dict = dict(zip(keys, values))

province_column = []
for i in census_poverty['province'].to_list():
    if i in list(conv_dict.keys()):
        province_column.append(conv_dict[i])
    else:
        province_column.append(i)
census_poverty['geoprovince'] = province_column
census_poverty = census_poverty.drop([92], axis=0)
converter = {'DAVAO DEL NORTE' : 'DAVAO (DAVAO DEL NORTE)', 
             'ISABELA CITY':'ISABELA', 'MT. PROVINCE':'MOUNTAIN PROVINCE',
            'NCR' : 'METRO MANILA', 'NORTH COTABATO': 'COTABATO (NORTH COT.)',
            'SARANGGANI':'SARANGANI', 'TAWI': 'TAWI-TAWI', 
            'WESTERN SAMAR':'SAMAR (WESTERN SAMAR)'}

census_cols = [i.upper() for i in census_poverty['province'].to_list()]
for i in range(len(census_cols)):
    try:
        census_cols[i] = converter[census_cols[i]]
    except:
        census_cols[i] = census_cols[i]
census_poverty['province'] = census_cols
census_poverty['poverty_incidence'] = census_poverty['poverty_incidence'] / 100
```


```python
df_poverty_duterte = df_geo.merge(census_poverty, left_on='province', right_on='province').sort_values(by='DUTERTE, RODY (PDPLBN)', ascending=False).iloc[:10, :]
```


```python
colors = 5
cmap = 'Reds'
figsize = (15, 15)
ax = df_poverty_duterte.plot(column='poverty_incidence', cmap=cmap, figsize=figsize, 
                          scheme='equal_interval', k=colors, legend=True, legend_kwds={'loc':2})
ax.set_title('Poverty Incidence in Duterte Voting Provinces 2016 - Philippine Total: 16%');
```


![png](DMW%20Blog%20%231_files/DMW%20Blog%20%231_9_0.png)


Aside from Davao del Sur, which houses the metropolitan city of Davao City, all of the provinces that overwhelmingly voted for Duterte in 2016 were above the national poverty incidence rate of 16%, with provinces like Lanao del Sur and Sulu having an incidence of 66% and 50% respectively. This further pushes the populist message of Duterte as being a "common man" who can relate to the struggles of these provinces.


## President Duterte's Candidates in 2019
In the intervening 3 years of his presidency, how has his populist message affected the way that voters vote for public officials? This can be seen in the win rate of his candidates for Senate in 2019, where we can measure how many of the elected officials were part of the administrations party, indicating that they were endorsed by Duterte himself. 

The political party Hugpong ng Pagbabago (HGP) was created in 2018 by President Duterte's daughter, Sara Duterte, to be able to rally allies from different political parties into their umbrella party. These political parties include: PDP-Laban, Nacionalista Party, Lakas-CMD, Pwersa ng Masang Pilipino, Nationalist People's Coalition, Laban ng Demokratikong Pilipino, among others. 

In the 2019 National Senatorial Elections, <b>9 of the top 12</b> candidates were all part of the HGP ballot and were directly endorsed by, or are allies of, President Duterte.


```python
def any_election_result(contest_code):
    
    '''
    Returns a dataframe of all candidates in contest_code with aggregated votes
    per ER levels, sorted from highest to lowest.
    
    Parameters
    ----------
    contest_code : int
                 : contest code of the electoral contest, comes from 
                   nle2019/contests directory.
    
    Returns
    -------
    df_results : pandas DataFrame
               : dataframe of all candidates for the electoral contest sorted from
                 highest votes to lowest.
                 
    '''
    # extract all file names of election returns
    filenames = []
    with open('2019_filepaths.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            filenames.append(row)
    filenames = filenames[0]
    
    # creates a dictionary of all candidates for the contest_code
    with open('/mnt/data/public/elections/nle2019/contests/'+ str(contest_code) + '.json', 'r') as file:
        data = json.load(file)
    candidates = []
    for can in data['bos']:
        candidates.append(can['boc'])
    candidate_data = data['bos']

    votes_blank = [0] * len(candidates)

    vote_count = dict(zip(candidates, votes_blank))

    # aggregates votes in contest_code for each candidate over all electoral returns
    for er in filenames:
        with open(er, 'r') as file:
            data = json.load(file)['rs']

            for vote in data:
                if vote['cc'] == contest_code:
                    vote_count[vote['bo']] += vote['v']

    # creates a dataframe of candidates in contest_code
    df_contest = pd.DataFrame(data=list(vote_count.values()), index=list(vote_count.keys()))
    df_contest.reset_index(inplace=True)
    df_contest.columns = ['candidate_code', 'votes']

    # merges dataframe of vote counts with candidates
    df_results = pd.DataFrame.from_dict(candidate_data)
    df_results = df_results.merge(df_contest, how='left', left_on='boc', right_on='candidate_code')
    
    return df_results.sort_values(by='votes', ascending=False)[['boc', 'bon', 'pn', 'votes']]
```


```python
df_senate = any_election_result(1)
```


```python
plot_senate = df_senate[['bon', 'votes']].iloc[:12,:]
```


```python
plot_senate.iplot(kind='bar', x='bon', y='votes', title='Top 12 Senators 2019')
```


<div>


            <div id="6adb576c-825f-4301-802f-f8e29bc5ae88" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';

                if (document.getElementById("6adb576c-825f-4301-802f-f8e29bc5ae88")) {
                    Plotly.newPlot(
                        '6adb576c-825f-4301-802f-f8e29bc5ae88',
                        [{"marker": {"color": "rgba(255, 153, 51, 0.6)", "line": {"color": "rgba(255, 153, 51, 1.0)", "width": 1}}, "name": "votes", "orientation": "v", "text": "", "type": "bar", "uid": "31106a8c-17b2-4ed5-9016-2968b7ac7e11", "x": ["VILLAR, CYNTHIA (NP)", "POE, GRACE (IND)", "GO, BONG GO (PDPLBN)", "CAYETANO, PIA (NP)", "DELA ROSA, BATO (PDPLBN)", "ANGARA, EDGARDO SONNY (LDP)", "LAPID, LITO (NPC)", "MARCOS, IMEE (NP)", "TOLENTINO, FRANCIS (PDPLBN)", "PIMENTEL, KOKO (PDPLBN)", "BONG REVILLA, RAMON JR (LAKAS)", "BINAY, NANCY (UNA)"], "y": [25119270, 21913372, 20456891, 19635682, 18805968, 18043625, 16881719, 15732733, 15370672, 14550540, 14547352, 14449334]}],
                        {"legend": {"bgcolor": "#F5F6F9", "font": {"color": "#4D5663"}}, "paper_bgcolor": "#F5F6F9", "plot_bgcolor": "#F5F6F9", "title": {"font": {"color": "#4D5663"}, "text": "Top 12 Senators 2019"}, "xaxis": {"gridcolor": "#E1E5ED", "showgrid": true, "tickfont": {"color": "#4D5663"}, "title": {"font": {"color": "#4D5663"}, "text": ""}, "zerolinecolor": "#E1E5ED"}, "yaxis": {"gridcolor": "#E1E5ED", "showgrid": true, "tickfont": {"color": "#4D5663"}, "title": {"font": {"color": "#4D5663"}, "text": ""}, "zerolinecolor": "#E1E5ED"}},
                        {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){

var gd = document.getElementById('6adb576c-825f-4301-802f-f8e29bc5ae88');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }};
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


Among the 12 Senators, 9 of them were allied with Duterte, with only Grace Poe (Independent), Lito Lapid (NPC), and Nancy Binay (UNA), not belonging to the list of Duterte allies.

In the provincial gubernatorial race, <b>69 out of 81</b> provincial governors who won were also allied with HGP or PDP-Laban.


```python
HGP = ['PARTIDO DEMOKRATIKO PILIPINO LAKAS NG BAYAN', 'NACIONALISTA PARTY', 
       'LAKAS CHRISTIAN  MUSLIM DEMOCRATS', 'HUGPONG NG PAGBABAGO PARTY',
      "NATIONALIST PEOPLE'S COALITION", 'NATIONAL UNITY PARTY']
```


```python
def governor_elections():
    results = []
    
    # contest codes for all gubernatorial contest
    contest_code = [i for i in range(2,83)]
    
    # get filepaths for all election returns
    filenames = []
    with open('2019_filepaths.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            filenames.append(row)
    filenames = filenames[0]
    
    # get all regions and provinces into a dictionary of region:province keys and values
    regions = glob.glob('/mnt/data/public/elections/nle2019/results/*')
    regions.remove('/mnt/data/public/elections/nle2019/results/info.json')
    regions = [re.findall(r'\/\w*\/\w*\/\w*\/\w*\/\w*\/'
    '\w*\/(.*)', i)[0] for i in regions]
    
    regional = []
    provinces = []
    for region in regions:
        prov = glob.glob('/mnt/data/public/elections/nle2019/'
                         'results/' + region + '/*')
        prov.remove('/mnt/data/public/elections/nle2019/'
                    'results/' + region + '/info.json')
        prov = [re.findall(r'\/mnt\/data\/public\/elections\/nle2019\/results'
                           '\/[\w\s\-]*\/(.*)', i)[0] for i in prov]
        regional.extend([region] * len(prov))
        provinces.extend(prov)
        
    prov_reg_dict = dict(zip(provinces, regional))
    
    for i in range(2,83):
        contest_code = i
        
        # get all candidates and create a dictionary for votes
        with open('/mnt/data/public/elections/nle2019/'
                  'contests/'+ str(contest_code) + '.json', 'r') as file:
            data = json.load(file)
            
        candidates = []
        for can in data['bos']:
            candidates.append(can['boc'])
        candidate_data = data['bos']

        votes_blank = [0] * len(candidates)

        vote_count = dict(zip(candidates, votes_blank))
        
        province = re.findall('PROVINCIAL GOVERNOR (.*)', data['cn'])[0]
        
        if province == 'DAVAO  (DAVAO DEL NORTE)':
            province = 'DAVAO (DAVAO DEL NORTE)'
            
        region = prov_reg_dict[province]

        # aggregates votes in contest_code for each candidate over all electoral returns
        escape_provinces = ['DAVAO (DAVAO DEL NORTE)', 
                            'SAMAR (WESTERN SAMAR)', 'COTABATO (NORTH COT.)']
        regex_provinces = ['DAVAO \(DAVAO DEL NORTE\)', 
                           'SAMAR \(WESTERN SAMAR\)', 
                           'COTABATO \(NORTH COT\.\)']
        prov_convert = dict(zip(escape_provinces, regex_provinces))
       
        if province in escape_provinces:
            prov_file = prov_convert[province]
        else:
            prov_file = province
        
        filenames_prov = [re.findall('/mnt/data/public/elections/nle2019/results/' + region + '/' + prov_file + '.*', x) for x in filenames]
        filenames_prov = [x[0] for x in filenames_prov if len(x) > 0]
        
        for er in filenames_prov:
            with open(er, 'r') as file:
                data = json.load(file)['rs']

                for vote in data:
                    if vote['cc'] == contest_code:
                        vote_count[vote['bo']] += vote['v']

        winner = max(vote_count.items(), key=operator.itemgetter(1))
        governor = {}
        governor['region'] = region
        governor['province'] = province
        with open('/mnt/data/public/elections/nle2019/contests/'+ str(contest_code) + '.json', 'r') as file:
            data = json.load(file)['bos']
        for i in data:
            if i['boc'] == winner[0]:
                governor['winner'] = i['bon']
                governor['political_party'] = i['pn']
        governor['votes'] = winner[1]
        
        results.append(governor)
        
    return pd.DataFrame.from_records(results)
```


```python
gov_race = governor_elections()
```


```python
gov_race_duterte_allies = gov_race.query('political_party in {}'.format(HGP))
```


```python
fig = {
  "data": [
    {
      "values": [69, 12],
      "labels": [
        "Duterte Affiliated",
        "Non-Duterte Affiliated",
    
      ],
        'marker': {
      'colors': [
        'rgb(186, 47, 41)',
          'rgb(243,189,165)'
      ]
    },
      "domain": {"column": 0},
      "name": "Affiliation",
      "hoverinfo":"label+value+name",
      "hole": .5,
      "type": "pie"
    },
    ],
  "layout": {
        "title":"Provincial Governor Winners 2019",
        "grid": {"rows": 1, "columns": 1},
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                
            },
            
        ]
    }
}
pyoff.iplot(fig, filename='donut')
```


<div>


            <div id="fdfbcec6-feb0-4add-9885-57c40018be0d" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';

                if (document.getElementById("fdfbcec6-feb0-4add-9885-57c40018be0d")) {
                    Plotly.newPlot(
                        'fdfbcec6-feb0-4add-9885-57c40018be0d',
                        [{"domain": {"column": 0}, "hole": 0.5, "hoverinfo": "label+value+name", "labels": ["Duterte Affiliated", "Non-Duterte Affiliated"], "marker": {"colors": ["rgb(186, 47, 41)", "rgb(243,189,165)"]}, "name": "Affiliation", "type": "pie", "uid": "2c14d39e-b74f-4e28-8c9d-7855f212e696", "values": [69, 12]}],
                        {"annotations": [{"font": {"size": 20}}], "grid": {"columns": 1, "rows": 1}, "title": {"text": "Provincial Governor Winners 2019"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){

var gd = document.getElementById('fdfbcec6-feb0-4add-9885-57c40018be0d');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


# Electoral Turnout 2019 and Voter Education

Duterte's allies won overwhelmingly in the 2019 National Elections for both Senatorial and Gubernatorial Races, so does this mean that the population of the Philippines is overwhelmingly in favor of President Duterte? To validate this, we look into the voter turnout for 2019 per province to find out what percent of total voters actually voted during the elections.


```python
def region_stats():
    
    '''
    Returns a dataframe of all voter statistics from each province and regions 
    precinct returns.

    Returns
    -------
    df : Pandas DataFrame
       : DataFrame containing all statistics per region and province aggregate from precinct returns
    '''
    
    results = []
    
    # contest codes for all gubernatorial contest
    contest_code = [i for i in range(2,83)]
    
    # get filepaths for all election returns
    filenames = []
    with open('2019_filepaths.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            filenames.append(row)
    filenames = filenames[0]
    
    # get all regions and provinces into a dictionary of region:province keys and values
    regions = glob.glob('/mnt/data/public/elections/nle2019/results/*')
    regions.remove('/mnt/data/public/elections/nle2019/results/info.json')
    regions = [re.findall(r'\/\w*\/\w*\/\w*\/\w*\/\w*\/\w*\/(.*)', i)[0] for i in regions]
    
    regional = []
    provinces = []
    for region in regions:
        prov = glob.glob('/mnt/data/public/elections/nle2019/results/' + region + '/*')
        prov.remove('/mnt/data/public/elections/nle2019/results/' + region + '/info.json')
        prov = [re.findall(r'\/mnt\/data\/public\/elections\/nle2019\/results\/[\w\s\-]*\/(.*)', i)[0] for i in prov]
        regional.extend([region] * len(prov))
        provinces.extend(prov)
        
    prov_reg_dict = dict(zip(provinces, regional))

    for i in range(2,83):
        contest_code = i
        
        # get all candidates and create a dictionary for votes
        with open(filenames[0], 'r') as file:
            data = json.load(file)
        
        columns = []   
        for j in data['cos']:
            if j['cc'] == 1:
                columns.append(j['cn'])
        
        columns.append('region')
        columns.append('province')
        
        sum_dict = {}
        for column in columns:
            sum_dict[column] = 0
        
        with open('/mnt/data/public/elections/nle2019/contests/' + str(i) + '.json', 'r') as file:
            data = json.load(file)
            
        province = re.findall('PROVINCIAL GOVERNOR (.*)', data['cn'])[0]
        
        if province == 'DAVAO  (DAVAO DEL NORTE)':
            province = 'DAVAO (DAVAO DEL NORTE)'
            
        region = prov_reg_dict[province]
        
        sum_dict['region'] = region
        sum_dict['province'] = province
        
        escape_provinces = ['DAVAO (DAVAO DEL NORTE)', 'SAMAR (WESTERN SAMAR)', 'COTABATO (NORTH COT.)']
        regex_provinces = ['DAVAO \(DAVAO DEL NORTE\)', 'SAMAR \(WESTERN SAMAR\)', 'COTABATO \(NORTH COT\.\)']
        prov_convert = dict(zip(escape_provinces, regex_provinces))
       
        if province in escape_provinces:
            prov_file = prov_convert[province]
        else:
            prov_file = province
            
        # aggregates votes in contest_code for each candidate over all electoral returns
        filenames_prov = [re.findall('/mnt/data/public/elections/nle2019/results/' + region + '/' + prov_file + '.*', x) for x in filenames]
        filenames_prov = [x[0] for x in filenames_prov if len(x) > 0]
        
        for er in filenames_prov:
            with open(er, 'r') as file:
                data = json.load(file)['cos']

            for k in data:
                for column in columns:
                    if k['cc'] == 1 and k['cn'] == column:
                        sum_dict[column] += k['ct']
        

        results.append(sum_dict)
    
    filepaths_ncr = glob.glob('/mnt/data/public/elections/nle2019/results/NCR/*/*/*/*')
    filepaths_ncr = [i for i in filepaths_ncr if i[-9:] != 'info.json']

    with open(filepaths_ncr[0], 'r') as file:
        data = json.load(file)

    columns = []   
    for j in data['cos']:
        if j['cc'] == 1:
            columns.append(j['cn'])

    columns.append('region')
    columns.append('province')

    sum_dict = {}
    for column in columns:
        sum_dict[column] = 0

    sum_dict['region'] = 'NCR'
    sum_dict['province'] = 'METRO MANILA'
    
    for ncr in filepaths_ncr:

        with open(ncr, 'r') as file:
            data = json.load(file)['cos']

        for line in data:
            for column in columns:
                if line['cc'] == 1 and line['cn'] == column:
                    sum_dict[column] += line['ct']

    results.append(sum_dict)
        
    df = pd.DataFrame.from_records(results)
    df.groupby('region')[['expected-voters', 'number-of-voters-who-actually-voted']].sum()
    df['turnout'] = df['number-of-voters-who-actually-voted'] /df['expected-voters']
    return df
```


```python
df_prov = provincial_stats()
```


```python
def elementary_graduates():
    
    '''
    Returns a dataframe of the total population and total elementary graduates per province from the 
    2015 census.
    
    Returns
    -------
    df   :    Pandas DataFrame
         :    A Dataframe with rows as regions, and columns as the total population,
              and total elementary graduates per region.   
    '''
    
    filepaths = glob.glob('/mnt/data/public/census/*')
    filepaths = [i for i in filepaths if i.split('/')[-1][0] == '_']
    filepaths = [i for i in filepaths if i != '/mnt/data/public/census/_PHILIPPINES_Statistical Tables.xls']
    regions = []
    total_population = []
    elementary_population = []

    for i in filepaths:
        region = re.findall('\_(.*)\_', i)[0]
        df = pd.read_excel(i, skiprows=5, sheet_name='T11')
        census_columns11 = ['educational_attainment', 'Total', 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,
                            19, '20-24', '25-29', '30-34', '35 and over']
        df.columns = census_columns11
        df = df.iloc[:54, :]
        voting_pop = ['educational_attainment','Total', 15,16,17,18,
                            19, '20-24', '25-29', '30-34', '35 and over']
        total_pop = df[voting_pop].iloc[0,1]
        elementary = df[voting_pop].loc[[7,10,13,14,15,16],:].iloc[:,2:].sum().sum()

        regions.append(region)
        total_population.append(total_pop)
        elementary_population.append(elementary)

    df_literacy = pd.DataFrame(regions)
    df_literacy['total_population'] = total_population
    df_literacy['elem_population'] = elementary_population
    df_literacy.columns = ['region', 'total_population', 'elem_population']
    df_literacy['elementary_rate'] = df_literacy['elem_population'] / df_literacy['total_population']

    return df_literacy
```


```python
df_elem = elementary_graduates()
```


```python
df_prov_agg = df_prov.groupby('region')[['number-of-voters-who-actually-voted', 'expected-voters']].sum()
df_prov_agg['turnout'] = df_prov_agg['number-of-voters-who-actually-voted'] / df_prov_agg['expected-voters']
df_prov_agg.reset_index(inplace=True)
```


```python
region_shape = geopandas.GeoDataFrame.from_file("./Geopandas Shapes/Regions.shp")
```


```python
elem_region_convert = ['Metropolitan Manila', 'Caraga (Region XIII)', 
                       'Cordillera Administrative Region (CAR)', 
                       'CALABARZON (Region IV-A)',
'Central Visayas (Region VII)', 'Northern Mindanao (Region X)',  
                       'Zamboanga Peninsula (Region IX)',
'Autonomous Region of Muslim Mindanao (ARMM)', 'MIMAROPA (Region IV-B)',
'Eastern Visayas (Region VIII)', 'Ilocos Region (Region I)',
'Bicol Region (Region V)', 'Cagayan Valley (Region II)', ' ',
'Central Luzon (Region III)', 'Davao Region (Region XI)',
'Western Visayas (Region VI)','SOCCSKSARGEN (Region XII)']
```


```python
prov_converter = ['Autonomous Region of Muslim Mindanao (ARMM)',
              'Cordillera Administrative Region (CAR)',
                 'Metropolitan Manila', 'Ilocos Region (Region I)',
                 'Cagayan Valley (Region II)', 'Central Luzon (Region III)',
                 'CALABARZON (Region IV-A)', 'MIMAROPA (Region IV-B)',
                'Zamboanga Peninsula (Region IX)', 'Bicol Region (Region V)',
                'Western Visayas (Region VI)', 'Central Visayas (Region VII)',
                 'Eastern Visayas (Region VIII)', 'Northern Mindanao (Region X)',
                'Davao Region (Region XI)', 'SOCCSKSARGEN (Region XII)',
                 'Caraga (Region XIII)']
```


```python
df_elem['geo_region'] = elem_region_convert
df_prov_agg['geo_region'] = prov_converter
df_geo_turnout = region_shape.merge(df_prov_agg, left_on='REGION', right_on='geo_region')
df_geo_elem = region_shape.merge(df_elem, left_on='REGION', right_on='geo_region')
```


```python
df_prov_agg[['geo_region', 'turnout']].sort_values('turnout').iplot(kind='bar', x='geo_region', 
                                             y='turnout', title='Voter Turnout by Region, 2019')
```


<div>


            <div id="4759cd14-3c8a-4127-9a02-712b723e2628" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';

                if (document.getElementById("4759cd14-3c8a-4127-9a02-712b723e2628")) {
                    Plotly.newPlot(
                        '4759cd14-3c8a-4127-9a02-712b723e2628',
                        [{"marker": {"color": "rgba(255, 153, 51, 0.6)", "line": {"color": "rgba(255, 153, 51, 1.0)", "width": 1}}, "name": "turnout", "orientation": "v", "text": "", "type": "bar", "uid": "83cb466b-2c7c-49fb-9ac3-a9d44a89913d", "x": ["Metropolitan Manila", "CALABARZON (Region IV-A)", "Davao Region (Region XI)", "Autonomous Region of Muslim Mindanao (ARMM)", "Zamboanga Peninsula (Region IX)", "SOCCSKSARGEN (Region XII)", "MIMAROPA (Region IV-B)", "Central Luzon (Region III)", "Cagayan Valley (Region II)", "Western Visayas (Region VI)", "Cordillera Administrative Region (CAR)", "Northern Mindanao (Region X)", "Central Visayas (Region VII)", "Bicol Region (Region V)", "Eastern Visayas (Region VIII)", "Caraga (Region XIII)", "Ilocos Region (Region I)"], "y": [0.6947474218572868, 0.7125136886272071, 0.7345201570081142, 0.7370635812231788, 0.7460662926640866, 0.749887815602166, 0.7600218699944226, 0.7709584254178036, 0.7728706943629781, 0.7800140168280987, 0.784918322348261, 0.7880227271453943, 0.79693584343859, 0.7972938186369963, 0.7983077354512558, 0.8067241400854944, 0.8093484514556779]}],
                        {"legend": {"bgcolor": "#F5F6F9", "font": {"color": "#4D5663"}}, "paper_bgcolor": "#F5F6F9", "plot_bgcolor": "#F5F6F9", "title": {"font": {"color": "#4D5663"}, "text": "Voter Turnout by Region, 2019"}, "xaxis": {"gridcolor": "#E1E5ED", "showgrid": true, "tickfont": {"color": "#4D5663"}, "title": {"font": {"color": "#4D5663"}, "text": ""}, "zerolinecolor": "#E1E5ED"}, "yaxis": {"gridcolor": "#E1E5ED", "showgrid": true, "tickfont": {"color": "#4D5663"}, "title": {"font": {"color": "#4D5663"}, "text": ""}, "zerolinecolor": "#E1E5ED"}},
                        {"showLink": true, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){

var gd = document.getElementById('4759cd14-3c8a-4127-9a02-712b723e2628');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


The top 10 provinces with the lowest voter turnout were:


```python
df_prov_agg.sort_values(by='turnout')[['turnout', 'geo_region']].loc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>turnout</th>
      <th>geo_region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.694747</td>
      <td>Metropolitan Manila</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.712514</td>
      <td>CALABARZON (Region IV-A)</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.734520</td>
      <td>Davao Region (Region XI)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.737064</td>
      <td>Autonomous Region of Muslim Mindanao (ARMM)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.746066</td>
      <td>Zamboanga Peninsula (Region IX)</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.749888</td>
      <td>SOCCSKSARGEN (Region XII)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.760022</td>
      <td>MIMAROPA (Region IV-B)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.770958</td>
      <td>Central Luzon (Region III)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.772871</td>
      <td>Cagayan Valley (Region II)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.780014</td>
      <td>Western Visayas (Region VI)</td>
    </tr>
  </tbody>
</table>
</div>



From this data, we can see that a majority of the Philippine population actually voted during the elections. Given this, we want to see the level of educational attainment for each province coming from the latest census. We will be using the number of elementary graduates per region as a substitute for the level of voter education and literacy, i.e. how informed are voters in these regions? 

Here is the breakdown of the % of elementary graduates per region:


```python
colors = 5
cmap = 'Blues'
figsize = (15, 15)
ax = df_geo_elem.plot(column='elementary_rate', cmap=cmap, figsize=figsize, 
                          scheme='equal_interval', k=colors, legend=True, legend_kwds={'loc':2})
ax.set_title('% of Elementary Graduates per Region');
```


![png](DMW%20Blog%20%231_files/DMW%20Blog%20%231_35_0.png)


For the entire population, the average elementary graduation rate is at 55%.


```python
ph_elementary_rate = round((df_elem['elem_population'].sum() / df_elem['total_population'].sum()), 2)
print('Philippines Elementary Graduation: ', ph_elementary_rate)
```

    Philippines Elementary Graduation:  0.55


This means that 55% of our total population can be inferred as being "informed voters" who are educated to be able to understand President Duterte's platform. Although, this number is ultimately inconclusive as there are a lot of other factors that come into play such as advertising spending, possible vote buying, etc. that could factor into the dominance of President Duterte's allies in the recent national elections.

## What does this mean for President Duterte's Administration and the voters' preferences?

By exploring the historic win of President Duterte in 2016 and looking into the 2019 elections, we see that the Duterte Administration has kept a firm hold on the population and has continued to get stronger and stronger. With a high percentage of electoral voter turnout in 2019, it can be inferred that the votes that ultimately went into the system were representative of the voting population. Through looking at the educational levels of each Region, we can see that the most informed/educated regions don't correlate with the votes that the Duterte administration received.

From the 2016 and 2019 data, we can see that President Duterte and his allies are only getting stronger in terms of the populat vote, as in 2019, 9/12 senators and 69/81 provincial governors elected were affiliated with President Duterte. 

However, there are other factors to look at that could affect this data, such as the incidence of vote-buying in each province, and the amount of advertising spending by each candidate that could ultimately lead to the higher votes for the Duterte Administration candidates.



# <center> Methodology </center>

### Presidential Elections 2016
In examining the presidential race from 2016, we looked into each precinct's JSON file located in the 2016 National Elections database. In order to filter through each file, we split the string of the absolute filepath in order to get only those that are marked "PRESIDENT PHILIPPINES.json", which pertains to the presidential elections. Each of these files contain the name of the presidential candidate inside a dictionary with the key 'bName', and their votes as 'votes'. Here is a sample of the loaded json file:


```python
with open('/mnt/data/public/elections/nle2016/PHILIPPINES/ARMM/BASILAN/'
          'ISABELA CITY/LANOTE/07010056/PRESIDENT PHILIPPINES.json', 'r') as file:
    data = json.load(file)['results']
data[0]
```




    {'bName': 'BINAY, JOJO (UNA)',
     'canCode': '89745',
     'percentage': '6.57',
     'votes': '28'}



We then created a list of candidates that would be used to create a dictionary per province level, that would in turn be used to aggregate votes for each candidate in the election. After aggregating these votes, we also passed in a key "region" and "province" that would take in the values of that specific region and province, to be used to create a Pandas DataFrame later on.

After aggregating all the data into a DataFrame, we converted the province names to match the province names in our geopandas DataFrame in order to be able to merge these together to create the choropleth with the shape files in the geopandas DataFrame.


```python

```

### Poverty Incidence
To be able to get the poverty index, we downloaded data from the Philippine Statistics Authority regarding the poverty incidence for each province. 

We first loaded the excel file into a pandas DataFrame and cleaned up all of the NaN values, after which we converted the province names to match the province names in our geopandas DataFrame for plotting later on. 

In order to get the percentage of poverty incidence, we divided the original values by 100.|


```python
# load census data with poverty incidence
census_poverty = pd.read_excel('PSA Census Poverty Incidence.xlsx', skiprows=2, usecols=['Unnamed: 0', 'Unnamed: 10'])
census_poverty.dropna(how='any', inplace=True)
census_poverty = census_poverty[census_poverty['Unnamed: 0'].str.startswith('..')]
census_poverty.loc[104] = ['NCR', 3.0]
census_poverty = census_poverty.iloc[4:,:]
cleanup_col = [re.findall('\.?\.?([\w\s\.]*)', i)[0] for i in census_poverty['Unnamed: 0'].to_list()]
for i in range(len(cleanup_col)):
    if cleanup_col[i][-2:] == ' b' or cleanup_col[i][-2:] == ' c':
        cleanup_col[i] = cleanup_col[i][:-2]
    else:
        pass
census_poverty['Unnamed: 0'] = cleanup_col
census_poverty.columns = ['province', 'poverty_incidence']
census_poverty.sort_values(by='province', inplace=True)
keys = []
for i in cleanup_col:
    if i.lower() in ph1.NAME_1.str.lower().to_list():
        pass
    else:
        keys.append(i)
values = ['Mountain Province', 'Samar', 'Isabela', 'Sarangani', 
         'North Cotabato', 'Tawi-Tawi', 'Metropolitan Manila']
conv_dict = dict(zip(keys, values))

province_column = []
for i in census_poverty['province'].to_list():
    if i in list(conv_dict.keys()):
        province_column.append(conv_dict[i])
    else:
        province_column.append(i)
census_poverty['geoprovince'] = province_column
census_poverty = census_poverty.drop([92], axis=0)
converter = {'DAVAO DEL NORTE' : 'DAVAO (DAVAO DEL NORTE)', 
             'ISABELA CITY':'ISABELA', 'MT. PROVINCE':'MOUNTAIN PROVINCE',
            'NCR' : 'METRO MANILA', 'NORTH COTABATO': 'COTABATO (NORTH COT.)',
            'SARANGGANI':'SARANGANI', 'TAWI': 'TAWI-TAWI', 
            'WESTERN SAMAR':'SAMAR (WESTERN SAMAR)'}

census_cols = [i.upper() for i in census_poverty['province'].to_list()]
for i in range(len(census_cols)):
    try:
        census_cols[i] = converter[census_cols[i]]
    except:
        census_cols[i] = census_cols[i]
census_poverty['province'] = census_cols
census_poverty['poverty_incidence'] = census_poverty['poverty_incidence'] / 100
```

### Senatorial Elections 2019
We created a function called `any_election_result` that would take in any `contest code` that can be found in the 2019 elections data under `nle2019/contests` path. This function takes in any contest code passed in and runs through all of the aggregated JSON files for each precinct's electoral returns, and matches the contest code with the contest code for the particular vote that we are trying to get. 

Each JSON file in the precinct returns had different contest codes and candidates and their votes as a nested dictionary. A sample can be found here:


```python
with open('/mnt/data/public/elections/nle2019/results/NCR/NATIONAL CAPITAL '
          'REGION - FOURTH DISTRICT/CITY OF MAKATI/'
          'BANGKAL/76020028.json', 'r') as file:
    data = json.load(file)['rs']
data[0]
```




    {'bo': 1,
     'cc': 1,
     'per': '0.09',
     'ser': 'HPM180PA01076279',
     'tot': 7251,
     'v': 7}



By creating a dictionary of candidates from the `nle2019/contests` directory, we were able to aggregate votes per province by referencing the candidate code and matching these with the number of votes received per candidate in each precinct's return.


```python
def any_election_result(contest_code):
    
    '''
    Returns a dataframe of all candidates in contest_code with aggregated votes
    per ER levels, sorted from highest to lowest.
    
    Parameters
    ----------
    contest_code : int
                 : contest code of the electoral contest, comes from 
                   nle2019/contests directory.
    
    Returns
    -------
    df_results : pandas DataFrame
               : dataframe of all candidates for the electoral contest sorted from
                 highest votes to lowest.
                 
    '''
    # extract all file names of election returns
    filenames = []
    with open('2019_filepaths.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            filenames.append(row)
    filenames = filenames[0]
    
    # creates a dictionary of all candidates for the contest_code
    with open('/mnt/data/public/elections/nle2019/contests/'+ str(contest_code) + '.json', 'r') as file:
        data = json.load(file)
    candidates = []
    for can in data['bos']:
        candidates.append(can['boc'])
    candidate_data = data['bos']

    votes_blank = [0] * len(candidates)

    vote_count = dict(zip(candidates, votes_blank))

    # aggregates votes in contest_code for each candidate over all electoral returns
    for er in filenames:
        with open(er, 'r') as file:
            data = json.load(file)['rs']

            for vote in data:
                if vote['cc'] == contest_code:
                    vote_count[vote['bo']] += vote['v']

    # creates a dataframe of candidates in contest_code
    df_contest = pd.DataFrame(data=list(vote_count.values()), index=list(vote_count.keys()))
    df_contest.reset_index(inplace=True)
    df_contest.columns = ['candidate_code', 'votes']

    # merges dataframe of vote counts with candidates
    df_results = pd.DataFrame.from_dict(candidate_data)
    df_results = df_results.merge(df_contest, how='left', left_on='boc', right_on='candidate_code')
    
    return df_results.sort_values(by='votes', ascending=False)[['boc', 'bon', 'pn', 'votes']]
```

### Provincial Governor Elections
To be able to get the governor elections, we looked into the contest codes of all gubernatorial election contests in 2019 that can be found in the JSON files under `nle2019/contests` directory. By being able to aggregate the codes, we iterated through them to get all the candidates running for each position. After iterating through these, we found all the filepaths for the relevant province using `glob.glob` method and matched the contest code to each candidates code, party name, and votes, and aggregated this into a dictionary of total votes per province. From this dictionary, we took the key and value of the highest votes, and appended this dictionary entry into a list.

Once we've aggregated through the entire list, we passed this into a DataFrame by using the `pd.DataFrame.from_records` method of Pandas to create our dataframe with each gubernatorial winner per candidate.


```python
def governor_elections():
    results = []
    
    # contest codes for all gubernatorial contest
    contest_code = [i for i in range(2,83)]
    
    # get filepaths for all election returns
    filenames = []
    with open('2019_filepaths.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            filenames.append(row)
    filenames = filenames[0]
    
    # get all regions and provinces into a dictionary of region:province keys and values
    regions = glob.glob('/mnt/data/public/elections/nle2019/results/*')
    regions.remove('/mnt/data/public/elections/nle2019/results/info.json')
    regions = [re.findall(r'\/\w*\/\w*\/\w*\/\w*\/\w*\/'
    '\w*\/(.*)', i)[0] for i in regions]
    
    regional = []
    provinces = []
    for region in regions:
        prov = glob.glob('/mnt/data/public/elections/nle2019/'
                         'results/' + region + '/*')
        prov.remove('/mnt/data/public/elections/nle2019/'
                    'results/' + region + '/info.json')
        prov = [re.findall(r'\/mnt\/data\/public\/elections\/nle2019\/results'
                           '\/[\w\s\-]*\/(.*)', i)[0] for i in prov]
        regional.extend([region] * len(prov))
        provinces.extend(prov)
        
    prov_reg_dict = dict(zip(provinces, regional))
    
    for i in range(2,83):
        contest_code = i
        
        # get all candidates and create a dictionary for votes
        with open('/mnt/data/public/elections/nle2019/'
                  'contests/'+ str(contest_code) + '.json', 'r') as file:
            data = json.load(file)
            
        candidates = []
        for can in data['bos']:
            candidates.append(can['boc'])
        candidate_data = data['bos']

        votes_blank = [0] * len(candidates)

        vote_count = dict(zip(candidates, votes_blank))
        
        province = re.findall('PROVINCIAL GOVERNOR (.*)', data['cn'])[0]
        
        if province == 'DAVAO  (DAVAO DEL NORTE)':
            province = 'DAVAO (DAVAO DEL NORTE)'
            
        region = prov_reg_dict[province]

        # aggregates votes in contest_code for each candidate over all electoral returns
        escape_provinces = ['DAVAO (DAVAO DEL NORTE)', 
                            'SAMAR (WESTERN SAMAR)', 'COTABATO (NORTH COT.)']
        regex_provinces = ['DAVAO \(DAVAO DEL NORTE\)', 
                           'SAMAR \(WESTERN SAMAR\)', 
                           'COTABATO \(NORTH COT\.\)']
        prov_convert = dict(zip(escape_provinces, regex_provinces))
       
        if province in escape_provinces:
            prov_file = prov_convert[province]
        else:
            prov_file = province
        
        filenames_prov = [re.findall('/mnt/data/public/elections/nle2019/results/' + region + '/' + prov_file + '.*', x) for x in filenames]
        filenames_prov = [x[0] for x in filenames_prov if len(x) > 0]
        
        for er in filenames_prov:
            with open(er, 'r') as file:
                data = json.load(file)['rs']

                for vote in data:
                    if vote['cc'] == contest_code:
                        vote_count[vote['bo']] += vote['v']

        winner = max(vote_count.items(), key=operator.itemgetter(1))
        governor = {}
        governor['region'] = region
        governor['province'] = province
        with open('/mnt/data/public/elections/nle2019/contests/'+ str(contest_code) + '.json', 'r') as file:
            data = json.load(file)['bos']
        for i in data:
            if i['boc'] == winner[0]:
                governor['winner'] = i['bon']
                governor['political_party'] = i['pn']
        governor['votes'] = winner[1]
        
        results.append(governor)
        
    return pd.DataFrame.from_records(results)
```

### Elections Voter Turnout 2019
To be able to see the turnout of voters for the 2019 Elections, we looked into each precicnt's JSON file for the voter statistics for each precinct that will give us the data for the `number-of-voters-who-actually-voted` and `expected-voters`.

Below is a sample of the JSON file:


```python
with open('/mnt/data/public/elections/nle2019/results/NCR/NATIONAL CAPITAL '
          'REGION - FOURTH DISTRICT/CITY OF MAKATI/'
          'BANGKAL/76020028.json', 'r') as file:
    data = json.load(file)['cos']
data[:16]
```




    [{'cc': 1, 'cn': 'blank', 'ct': 4},
     {'cc': 1, 'cn': 'blank-ballots', 'ct': 0},
     {'cc': 1, 'cn': 'expected-voters', 'ct': 997},
     {'cc': 1, 'cn': 'misfeed-ballots', 'ct': 17},
     {'cc': 1, 'cn': 'number-of-voters-who-actually-voted', 'ct': 685},
     {'cc': 1, 'cn': 'overvote-ballots', 'ct': 44},
     {'cc': 1, 'cn': 'overvote-count', 'ct': 15},
     {'cc': 1, 'cn': 'overvotes', 'ct': 35},
     {'cc': 1, 'cn': 'previously-scanned-ballots', 'ct': 8},
     {'cc': 1, 'cn': 'processed-ballots', 'ct': 685},
     {'cc': 1, 'cn': 'processed-count', 'ct': 685},
     {'cc': 1, 'cn': 'resets', 'ct': 2},
     {'cc': 1, 'cn': 'returned-ballots', 'ct': 25},
     {'cc': 1, 'cn': 'turn-out', 'ct': 68.7},
     {'cc': 1, 'cn': 'under-votes', 'ct': 789},
     {'cc': 1, 'cn': 'undervote-ballots', 'ct': 356}]



By applying the same methodology as the previous codes, we were able to get a dictionary with the keys as the "statistic" that we were looking at, as well as the values for each precinct's return. By looping through each province, we were able to get the aggregated voter statistics per province and per region.


```python
def region_stats():
    '''
    Returns a dataframe of all voter statistics from each province and regions 
    precinct returns.

    Returns
    -------
    df : Pandas DataFrame
       : DataFrame containing all statistics per region and province aggregate from precinct returns
    '''
    
    results = []
    
    # contest codes for all gubernatorial contest
    contest_code = [i for i in range(2,83)]
    
    # get filepaths for all election returns
    filenames = []
    with open('2019_filepaths.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            filenames.append(row)
    filenames = filenames[0]
    
    # get all regions and provinces into a dictionary of region:province keys and values
    regions = glob.glob('/mnt/data/public/elections/nle2019/results/*')
    regions.remove('/mnt/data/public/elections/nle2019/results/info.json')
    regions = [re.findall(r'\/\w*\/\w*\/\w*\/\w*\/\w*\/\w*\/(.*)', i)[0] for i in regions]
    
    regional = []
    provinces = []
    for region in regions:
        prov = glob.glob('/mnt/data/public/elections/nle2019/results/' + region + '/*')
        prov.remove('/mnt/data/public/elections/nle2019/results/' + region + '/info.json')
        prov = [re.findall(r'\/mnt\/data\/public\/elections\/nle2019\/results\/[\w\s\-]*\/(.*)', i)[0] for i in prov]
        regional.extend([region] * len(prov))
        provinces.extend(prov)
        
    prov_reg_dict = dict(zip(provinces, regional))

    for i in range(2,83):
        contest_code = i
        
        # get all candidates and create a dictionary for votes
        with open(filenames[0], 'r') as file:
            data = json.load(file)
        
        columns = []   
        for j in data['cos']:
            if j['cc'] == 1:
                columns.append(j['cn'])
        
        columns.append('region')
        columns.append('province')
        
        sum_dict = {}
        for column in columns:
            sum_dict[column] = 0
        
        with open('/mnt/data/public/elections/nle2019/contests/' + str(i) + '.json', 'r') as file:
            data = json.load(file)
            
        province = re.findall('PROVINCIAL GOVERNOR (.*)', data['cn'])[0]
        
        if province == 'DAVAO  (DAVAO DEL NORTE)':
            province = 'DAVAO (DAVAO DEL NORTE)'
            
        region = prov_reg_dict[province]
        
        sum_dict['region'] = region
        sum_dict['province'] = province
        
        escape_provinces = ['DAVAO (DAVAO DEL NORTE)', 'SAMAR (WESTERN SAMAR)', 'COTABATO (NORTH COT.)']
        regex_provinces = ['DAVAO \(DAVAO DEL NORTE\)', 'SAMAR \(WESTERN SAMAR\)', 'COTABATO \(NORTH COT\.\)']
        prov_convert = dict(zip(escape_provinces, regex_provinces))
       
        if province in escape_provinces:
            prov_file = prov_convert[province]
        else:
            prov_file = province
            
        # aggregates votes in contest_code for each candidate over all electoral returns
        filenames_prov = [re.findall('/mnt/data/public/elections/nle2019/results/' + region + '/' + prov_file + '.*', x) for x in filenames]
        filenames_prov = [x[0] for x in filenames_prov if len(x) > 0]
        
        for er in filenames_prov:
            with open(er, 'r') as file:
                data = json.load(file)['cos']

            for k in data:
                for column in columns:
                    if k['cc'] == 1 and k['cn'] == column:
                        sum_dict[column] += k['ct']
        

        results.append(sum_dict)
    
    filepaths_ncr = glob.glob('/mnt/data/public/elections/nle2019/results/NCR/*/*/*/*')
    filepaths_ncr = [i for i in filepaths_ncr if i[-9:] != 'info.json']

    with open(filepaths_ncr[0], 'r') as file:
        data = json.load(file)

    columns = []   
    for j in data['cos']:
        if j['cc'] == 1:
            columns.append(j['cn'])

    columns.append('region')
    columns.append('province')

    sum_dict = {}
    for column in columns:
        sum_dict[column] = 0

    sum_dict['region'] = 'NCR'
    sum_dict['province'] = 'METRO MANILA'
    
    for ncr in filepaths_ncr:

        with open(ncr, 'r') as file:
            data = json.load(file)['cos']

        for line in data:
            for column in columns:
                if line['cc'] == 1 and line['cn'] == column:
                    sum_dict[column] += line['ct']

    results.append(sum_dict)
        
    df = pd.DataFrame.from_records(results)
    df.groupby('region')[['expected-voters', 'number-of-voters-who-actually-voted']].sum()
    df['turnout'] = df['number-of-voters-who-actually-voted'] /df['expected-voters']
    return df
```

### Percent of Elementary Graduates per Region

In order to see the level of informed-ness of each Region when it comes to voting, we looked at the percent of educational attainment pertaining to elementary graduates per Region. 

This came from the 2015 Census data, aggregated by Region and selected by filtering each Region's population by elementary graduates and above.


```python
def elementary_graduates():
    
    '''
    Returns a dataframe of the total population and total elementary graduates per province from the 
    2015 census.
    
    Returns
    -------
    df   :    Pandas DataFrame
         :    A Dataframe with rows as regions, and columns as the total population,
              and total elementary graduates per region.   
    '''
    
    filepaths = glob.glob('/mnt/data/public/census/*')
    filepaths = [i for i in filepaths if i.split('/')[-1][0] == '_']
    filepaths = [i for i in filepaths if i != '/mnt/data/public/census/_PHILIPPINES_Statistical Tables.xls']
    regions = []
    total_population = []
    elementary_population = []

    for i in filepaths:
        region = re.findall('\_(.*)\_', i)[0]
        df = pd.read_excel(i, skiprows=5, sheet_name='T11')
        census_columns11 = ['educational_attainment', 'Total', 5, 6, 7,8,9,10,11,12,13,14,15,16,17,18,
                            19, '20-24', '25-29', '30-34', '35 and over']
        df.columns = census_columns11
        df = df.iloc[:54, :]
        voting_pop = ['educational_attainment','Total', 15,16,17,18,
                            19, '20-24', '25-29', '30-34', '35 and over']
        total_pop = df[voting_pop].iloc[0,1]
        elementary = df[voting_pop].loc[[7,10,13,14,15,16],:].iloc[:,2:].sum().sum()

        regions.append(region)
        total_population.append(total_pop)
        elementary_population.append(elementary)

    df_literacy = pd.DataFrame(regions)
    df_literacy['total_population'] = total_population
    df_literacy['elem_population'] = elementary_population
    df_literacy.columns = ['region', 'total_population', 'elem_population']
    df_literacy['elementary_rate'] = df_literacy['elem_population'] / df_literacy['total_population']

    return df_literacy
```


```python

```
