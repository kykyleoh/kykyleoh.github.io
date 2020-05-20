---
title: "Unsupervised Clustering Analysis of NBA Players"
excerpt: "An analysis of the changes in the NBA landscape through the years using unsurpervised clustering methods.<br/><img src='/images/nba-analysis/nba-titlecard.png'>"
collection: portfolio
---


<h1>Overview</h1>
<p>This technical report was created for our Data Mining and Wrangling class in AIM MSDS. In particular, this was done during our 2nd semester of class, as one of the required lab reports. In this report, we sought to understand how the landscape of the NBA has changed over the decades, and specifically if we are able to generalize certain player stereotypes throughout the years. We analyze these stereotypes, as well as the changes among them, using Unsupervised Clustering and apply Principal Component Analysis to extract meaningful features from the data. At the end, we also take a look at the evolution of 3-point shooters and the dramatic change that the 3-point shot has introduced to the NBA gameplay (as part of my personal interest, mostly).</p>

<h2>Acknowledgements</h2>
<p>This analysis was done together with my Lab partner, Lance Aven Sy.</p>

## Imports and Functions

```python
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import seaborn as sns
import re

plt.style.use('https://gist.githubusercontent.com/lpsy/e81ff2c0decddc9c6df'
              'eb2fcffe37708/raw/lsy_personal.mplstyle')
```


```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from sklearn.metrics import calinski_harabaz_score, silhouette_score
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from collections import Counter

from warnings import simplefilter

simplefilter('ignore')
from collections import Counter
from wordcloud import WordCloud
```


```python
def cluster_range(X, clusterer, k_stop, actual=None):
    """Return a dictionary of cluster labels, internal validation values
    and, if actual labels is given, external validation values for every k
    starting from k = 2

    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
        Does not accept sparse matrices
    clusterer : array
        sklearn.cluster object
    k_stop : integer
        ending number of clusters
    actual : array, optional
        cluster labels

    Returns
    -------
    out : dict
    """
    out = {'chs': [], 'iidrs': [], 'inertias': [], 'scs': [], 'ys': []}
    if isinstance(actual, np.ndarray):
        out['amis'] = []
        out['ars'] = []
        out['ps'] = []

    for k in range(2, k_stop+1):
        clusterer.n_clusters = k

        y = clusterer.fit_predict(X)
        out['ys'].append(y)

        # Calinski-Harabasz index
        out['chs'].append(calinski_harabaz_score(X, y))

        # Intra/Inter cluster distance ratio
        out['iidrs'].append(intra_to_inter(X, y, euclidean, 50))

        # inertias
        out['inertias'].append(clusterer.inertia_)

        # Silhouette score
        out['scs'].append(silhouette_score(X, y))

        if isinstance(actual, np.ndarray):
            # Adjusted mutual information
            out['amis'].append(adjusted_mutual_info_score(
                actual, y, average_method='arithmetic'))

            # Adjusted Rand Index
            out['ars'].append(adjusted_rand_score(actual, y))

           # Cluster purity
            out['ps'].append(purity(actual, y))
    return out
```


```python
def plot_clusters(tsne_df, ys):
    n = len(ys)
    rows = int(round(np.sqrt(n)))
    cols = int(round(n/rows))

    if cols > rows:
        cols, rows = rows, cols

    fig, axes = plt.subplots(rows, cols, dpi=150, figsize=(rows*5+1,cols*4+1))
    
    for i, ax in enumerate(fig.axes):
        if i >= n:
            fig.delaxes(ax)
            continue
        ax.scatter(x='x', y='y', c=ys[i], alpha=0.8, data=tsne_df)
        ax.set_title(f'{i+2} clusters')
    
    return fig
```


```python
def intra_to_inter(X, y, dist, r):
    """Compute intracluster to intercluster distance ratio
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    r : integer
        Number of pairs to sample
        
    Returns
    -------
    ratio : float
        Intracluster to intercluster distance ratio
    """
    p = []
    q = []
    np.random.seed(11)
    for i, j in np.random.randint(low=0, high=len(y), size=(r, 2)): 
        if i == j:
            continue
        elif (y[i] == y[j]):
            p.append(dist(X[i],X[j]))
        else:
            q.append(dist(X[i],X[j]))
    return (np.asarray(p).mean())/(np.asarray(q).mean())
```


```python
def plot_internal(inertias, chs, iidrs, scs):
    """Plot internal validation values"""
    
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(2,2, figsize=(16,9))
    
    ks = np.arange(2, len(inertias)+2)
    ax[0][0].plot(ks, inertias, '-o', label='SSE', c=colors[0])
    ax[0][0].set_xlabel('$k$')
    ax[0][0].set_ylabel('SSE')
    ax[0][0].set_title('SSE')

    ax[1][0].plot(ks, chs, '-o', label='CH', c=colors[1])
    ax[1][0].set_xlabel('$k$')
    ax[1][0].set_ylabel('CH')
    ax[1][0].set_title('CH')
    ax[1][0]._get_lines.get_next_color()
    
    ax[0][1].plot(ks, iidrs, '-o', label='Inter-intra', c=colors[2])
    ax[0][1].set_xlabel('$k$')
    ax[0][1].set_ylabel('Inter-Intra') 
    ax[0][1].set_title('Inter-Intra')
    ax[0][1]._get_lines.get_next_color()
    
    ax[1][1].plot(ks, scs, '-o', label='Silhouette coefficient', c=colors[3])
    ax[1][1].set_xlabel('$k$')
    ax[1][1].set_ylabel('Silhouette') 
    ax[1][1].set_title('Silhouette')
    ax[1][1]._get_lines.get_next_color()
    
    for axs in fig.get_axes():
        for axis in [axs.xaxis, axs.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))   
    
    plt.tight_layout()
    return fig
```

# Exploratory Data Analysis (EDA)


```python
# load season averages
eda = pd.read_sql('''SELECT * FROM season_average''', conn)

# drop all empty or None rows
eda.dropna(how='any', inplace=True)

# remove all non-numeric data
eda = eda[~eda['G'].str.contains('Did')]

# convert all numeric columns to float
eda[eda.columns[6:-1]] = eda[eda.columns[6:-1]].astype(float)

# drop index columns
eda.drop('index', axis=1, inplace=True)
```

One of the most interesting developments in the NBA's recent history is the growing prevalence of 3 point shots. This was first popularized by the Steve Nash-led Phoenix Suns of 2004-2006, with their run-and-gun style offense under coach Mike D'Antoni. However, during the time, this was seen as more of a fad as the Phoenix Suns were never able to move past the Western Conference Finals and thus were not able to gain mainstream success. In today's NBA game, teams have utilized the 3 point shot to great effect; this is seen most in the Golden State Warriors who have won 3 of the last 5 championships behind their "Splash Brothers", Klay Thompson and 2-time MVP Steph Curry, and the Houston Rockets, who hold the record for the most 3 point attempts per game of a team in NBA history. With this in mind, let's take a look at the past 20 years worth of 3 point attempts and 3 point shooting percentage to see the growth of both volume and accuracy of the 3 point shot.


```python
seasons = eda.groupby('Season')[['3PA', 'PTS', '3P%']].mean().reset_index()

fig, ax = plt.subplots(figsize=(16,8), dpi=200)
ax.plot(seasons['Season'], seasons['3P%'], color='k', label='3 Point Percentage')
ax2 = ax.twinx()
ax2.plot(seasons['Season'], seasons['3PA'], color='green', label='3 Point Attempts')
ax.tick_params('x', labelrotation=75)
ax.legend()
ax2.legend(loc='upper left')
ax.axvline(0, color='green', ls='--', alpha=0.5)
ax.axvline(14, color='red', ls='--', alpha=0.5)
ax.axvline(17, color='red', ls='--', alpha=0.5)
ax.axvline(35, color='blue', ls='--', alpha=0.5)
ax.set_title('3 Point Attempts and Percentage 1979-2019');1
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_11_0.png)


In the plot above, we can see that during the introduction of the 3 point line <font color='green'>1979-80 Season</font>, 3 point accuracy was very high but this was limited to a very small sample size. The <font color='red'>red dotted line</font> during the 1994-95 and 1996-97 seasons indicate the 3-year period wherein the NBA shortened the 3 point line in order to increase volume and usage of the 3 pointer in the NBA game. Lastly, we can see the rapid increase in both the attempts and accuracy of the 3 pointer during the 2014-15 season onwards. This is marked by the <font color='blue'>blue</font> dotted line that indicates the year that Steph Curry won his first MVP season and the Golden State Warriors dominated the NBA to win their first championship in 50 years. This is a turning point in the 3 point arena of the game, as most teams in the current NBA cannot survive without a good 3 point shooter, and this is reflected in the marked increase in both volume and accuracy of 3 point shooters in the league since then. 

## Load Data

We proceed with clustering the NBA players.


```python
# connect to sqlite db
conn = sqlite3.connect('Lab_Lab 5_nbaDB.db')
```


```python
df = pd.read_sql('SELECT * FROM season_average', conn, index_col='index')
```


```python
df.head()
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
      <th>Season</th>
      <th>Age</th>
      <th>Tm</th>
      <th>Lg</th>
      <th>Pos</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>FG</th>
      <th>FGA</th>
      <th>...</th>
      <th>ORB</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
      <th>Player</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1990-91</td>
      <td>22.0</td>
      <td>POR</td>
      <td>NBA</td>
      <td>PF</td>
      <td>43.0</td>
      <td>0.0</td>
      <td>6.7</td>
      <td>1.3</td>
      <td>2.7</td>
      <td>...</td>
      <td>0.6</td>
      <td>1.4</td>
      <td>2.1</td>
      <td>0.3</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>0.5</td>
      <td>0.9</td>
      <td>3.1</td>
      <td>Alaa Abdelnaby</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1991-92</td>
      <td>23.0</td>
      <td>POR</td>
      <td>NBA</td>
      <td>PF</td>
      <td>71.0</td>
      <td>1.0</td>
      <td>13.2</td>
      <td>2.5</td>
      <td>5.1</td>
      <td>...</td>
      <td>1.1</td>
      <td>2.5</td>
      <td>3.7</td>
      <td>0.4</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>0.9</td>
      <td>1.9</td>
      <td>6.1</td>
      <td>Alaa Abdelnaby</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1992-93</td>
      <td>24.0</td>
      <td>TOT</td>
      <td>NBA</td>
      <td>PF</td>
      <td>75.0</td>
      <td>52.0</td>
      <td>17.5</td>
      <td>3.3</td>
      <td>6.3</td>
      <td>...</td>
      <td>1.7</td>
      <td>2.8</td>
      <td>4.5</td>
      <td>0.4</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>1.3</td>
      <td>2.5</td>
      <td>7.7</td>
      <td>Alaa Abdelnaby</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1992-93</td>
      <td>24.0</td>
      <td>MIL</td>
      <td>NBA</td>
      <td>PF</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>13.3</td>
      <td>2.2</td>
      <td>4.7</td>
      <td>...</td>
      <td>1.0</td>
      <td>2.1</td>
      <td>3.1</td>
      <td>0.8</td>
      <td>0.5</td>
      <td>0.3</td>
      <td>1.1</td>
      <td>2.0</td>
      <td>5.3</td>
      <td>Alaa Abdelnaby</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1992-93</td>
      <td>24.0</td>
      <td>BOS</td>
      <td>NBA</td>
      <td>PF</td>
      <td>63.0</td>
      <td>52.0</td>
      <td>18.3</td>
      <td>3.5</td>
      <td>6.6</td>
      <td>...</td>
      <td>1.8</td>
      <td>3.0</td>
      <td>4.8</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>1.3</td>
      <td>2.6</td>
      <td>8.2</td>
      <td>Alaa Abdelnaby</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



As the collected data for seasons are written in the NBA format of yyyy-yy, we create a function named `get_year` in order to create a new column that contains the ending year of each season i.e., 2008-09 will become 2009. 


```python
def get_year(x):
    year = re.search('\d+$', x).group(0)
    year = int(year)
    
    if year < 20:
        year += 2000
    else:
        year += 1900
        
    return year
```

As we are interested in seeing the progression of the NBA players for the last 10 years (2009-2019), we filter our dataframe to exclude all the years before the 2009 season. The resulting dataframe contains null values and non-numeric values i.e., "Did not play", which we remove by dropping these rows. Lastly, we slice our final dataframe columns to include only the columns that are not correlated as shown in the heatmap discussed previously.


```python
df['year'] = df.Season.apply(get_year)
df2 = df[df.year>=2009].drop('Season', axis=1)
df2.shape
```




    (7092, 31)




```python
df3 = df2.drop_duplicates(subset=['Player', 'year'])
df3 = df3.sort_values(['Player', 'year']).reset_index(drop=True)
print(df3.shape)
df3.head(2)
```

    (5550, 31)





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
      <th>Age</th>
      <th>Tm</th>
      <th>Lg</th>
      <th>Pos</th>
      <th>G</th>
      <th>GS</th>
      <th>MP</th>
      <th>FG</th>
      <th>FGA</th>
      <th>FG%</th>
      <th>...</th>
      <th>DRB</th>
      <th>TRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>TOV</th>
      <th>PF</th>
      <th>PTS</th>
      <th>Player</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>24.0</td>
      <td>DAL</td>
      <td>NBA</td>
      <td>C</td>
      <td>22</td>
      <td>0</td>
      <td>7.4</td>
      <td>0.8</td>
      <td>1.9</td>
      <td>0.405</td>
      <td>...</td>
      <td>1.3</td>
      <td>1.6</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>2.2</td>
      <td>A.J. Hammons</td>
      <td>2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.0</td>
      <td>IND</td>
      <td>NBA</td>
      <td>PG</td>
      <td>56.0</td>
      <td>2.0</td>
      <td>15.4</td>
      <td>2.6</td>
      <td>6.3</td>
      <td>0.41</td>
      <td>...</td>
      <td>1.4</td>
      <td>1.6</td>
      <td>1.9</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>1.1</td>
      <td>0.9</td>
      <td>7.3</td>
      <td>A.J. Price</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 31 columns</p>
</div>




```python
df3.columns
['Age', 'Pos', 'GS', 'MP', 'FG', 'TOV', 'PTS']
```




    Index(['Age', 'Tm', 'Lg', 'Pos', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P',
           '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB',
           'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Player',
           'year'],
          dtype='object')




```python
df4 = df3[df3['G'].apply(lambda x: x.replace('.', '').isnumeric())]
df4 = df4[['Player', 'year', 'G', '3P', '3P%', '2P', '2P%', 'FT%',
                'eFG%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF']]

df4.dropna(inplace=True)
```


```python
print(df4.shape)
df4.head(2)
```

    (4527, 15)





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
      <th>Player</th>
      <th>year</th>
      <th>G</th>
      <th>3P</th>
      <th>3P%</th>
      <th>2P</th>
      <th>2P%</th>
      <th>FT%</th>
      <th>eFG%</th>
      <th>ORB</th>
      <th>DRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>PF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A.J. Hammons</td>
      <td>2017</td>
      <td>22</td>
      <td>0.2</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.375</td>
      <td>0.45</td>
      <td>0.464</td>
      <td>0.4</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A.J. Price</td>
      <td>2010</td>
      <td>56.0</td>
      <td>1.1</td>
      <td>0.345</td>
      <td>1.5</td>
      <td>0.472</td>
      <td>0.8</td>
      <td>0.494</td>
      <td>0.2</td>
      <td>1.4</td>
      <td>1.9</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4['Player_Unique'] = df4.Player + '_' + df4.year.astype(str)
df4.loc[:, 'G':'PF'] = df4.loc[:,'G':'PF'].astype(np.float64)
```


```python
print(df4.shape)
df4.head(2)
```

    (4527, 16)





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
      <th>Player</th>
      <th>year</th>
      <th>G</th>
      <th>3P</th>
      <th>3P%</th>
      <th>2P</th>
      <th>2P%</th>
      <th>FT%</th>
      <th>eFG%</th>
      <th>ORB</th>
      <th>DRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>PF</th>
      <th>Player_Unique</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A.J. Hammons</td>
      <td>2017</td>
      <td>22.0</td>
      <td>0.2</td>
      <td>0.500</td>
      <td>0.5</td>
      <td>0.375</td>
      <td>0.45</td>
      <td>0.464</td>
      <td>0.4</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>A.J. Hammons_2017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A.J. Price</td>
      <td>2010</td>
      <td>56.0</td>
      <td>1.1</td>
      <td>0.345</td>
      <td>1.5</td>
      <td>0.472</td>
      <td>0.80</td>
      <td>0.494</td>
      <td>0.2</td>
      <td>1.4</td>
      <td>1.9</td>
      <td>0.6</td>
      <td>0.1</td>
      <td>0.9</td>
      <td>A.J. Price_2010</td>
    </tr>
  </tbody>
</table>
</div>



In the final dataframe, we created a new column that concatenates the player names with the season played, as there will be multiple years per player in the dataset. We are left with the following columns:
* `G` - Number of games played
* `3P` - Number of 3 point shots made
* `3P%` - Percentage of 3 point shots made
* `2P` - Number of 2 point shots made
* `2P%` - Percentage of 2 point shots made
* `FT%` - Percentage of free throws made
* `eFG%` - Effective field goal percentage ((FGM + 0.5 * 3PM) / FGA)
* `ORB` - Number of Offensive Rebounds
* `DRB` - Number of Defensive Rebounds
* `AST` - Number of assists
* `STL` - Number of steals
* `BLK` - Number of blocks
* `PF` - Number of personal fouls

Total number of features: 13.

## Dimensionality Reduction (PCA)

As we are still left with 13 features, we perform dimensionality reduction using Principal Component Analysis. Using PCA, we transform the dataset on to the vectors that have the most explained variance. As such, each principal component in the analysis is not representative of a feature, but rather, a combination of all features that are weighted. In doing so, we are able to reduce the number of "features" that we will be using for the analysis, thereby reducing the amount of noise in the model. 

However, before proceeding to Principal Component Analysis, we must first ensure that the dataset is mean-centered at 0 as this is one of the pre-requisites of PCA. We can achieve this by using the `StandardScaler` function from `sklearn.preprocessing`. This function scales all the features in order to have a new mean centered at 0. 


```python
X = df4.iloc[:,2:-1]
```


```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Once we have scaled the dataset, we take a look at the directions of each of the vectors that represent a feature in our data. This will give us a better idea in terms of the directionality or relationship of each feature with each other. In the plot below, we show the direction of each feature vector (<font color='red'>red arrow</font>) on the plot of the first 2 principal components.


```python
V = np.cov(X_scaled, rowvar=False)
lambdas, w = np.linalg.eig(V)
indices = np.argsort(lambdas)[::-1]
lambdas = lambdas[indices]
w = w[:, indices]
new_X = np.dot(X_scaled, w)
```


```python
fig, ax = plt.subplots(dpi=120)
ax.scatter(new_X[:,0], new_X[:,1])
for feat, feat_name in zip(w, df4.columns[2:-1]):
    ax.arrow(0, 0, 7*feat[0], 7*feat[1], color='r', width=0.1, ec='none')
    ax.text(7*feat[0], 7*feat[1], feat_name, ha='center', color='k')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Plot of Eigenvectors against first 2 PCs');
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_35_0.png)


In the plot of the feature vectors above, we can see that there are relationships among the feature vectors that are of interest in the context of an NBA game:

* The 3 point shooting vector (as indicated by `3P` and `3P%`) is almost opposite to the feature vector for offensive rebounds and blocks (as indicated by `ORB` and `BLK`). In the context of an NBA game, this makes sense as a player who shoots three pointers would most likely be positioned outside of the 3 point line, thus giving him a disadvantage on offensive rebounds due to the distance from the basket. This accounts for the negative relationship between these two vectors.
* In the same vein, 2 point shooting is correlated with defensive rebounds, personal fouls, and effective field goal percentage (as indicated by `DRB`, `PF`, and `eFG%`, respectively). This is most likely due to the "inside play" during NBA games, and can be seen the most among centers and power forwards as these positions are typically played near the paint. This proximity to the rim accounts for their high effective field goal percentage, propensity for defensive rebounds, and their prevalence of 2 point shots. As these players are also the ones most likely to be inside the paint, they are correlated with personal fouls as well as they are tasked with defending the rim from opponents, thus running straight into the line of fire of driving opponents and increasing their probability of committing a foul.
* An interesting observation is that that in the traditional positions of the NBA, we can see a separation along the vector of `eFG%` wherein all the vectors above it (`2P`, `PF`, `DRB`, `2P%`, `ORB`, and `BLK`) are traditionally attributes of Centers and Forwards, or the "inside" players. Whereas the vectors opposite of this group (`G`, `STL`, `AST`, `FT%`, `3P`, `3P%`) are traditionally attributed more toward guards who are the "outside" players and ball handlers of the game.

A main advantage of PCA is being able to limit the number of features that we use to describe the dataset. To achieve this, we will need to limit the original features by setting a target percentage (%) of explained variance that we want to retain using the lowest number of principal components. The functions below convert our scaled numpy array into the principal components and remove those principal components that are beyond our target explained variance of 80%. This leaves us with 5 principal components as can be seen in the cumulative variance explained plot below.


```python
def get_min_pcs(X, var):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    pca = PCA(svd_solver='full')
    new_X2 = pca.fit_transform(X)
    
    var_explained = pca.explained_variance_ratio_
    
    fig, ax = plt.subplots(1, 2, figsize=(16,6))
    ax[0].plot(np.arange(1, len(var_explained)+1), var_explained, c=colors[0])
    ax[0].set_xlabel('PC')
    ax[0].set_ylabel('variance explained')
    
    cum_var_explained = var_explained.cumsum()
    ax[1].plot(np.arange(1, len(cum_var_explained)+1),
                  cum_var_explained, '-o', c=colors[1])
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlabel('PC')
    ax[1].set_ylabel('cumulative variance explained');
    
    return new_X2, np.searchsorted(cum_var_explained, var) + 1

def project(X_rotated, min_pcs):
    pca = PCA(n_components=min_pcs, svd_solver='full')
    X_new = pca.fit_transform(X_rotated)
    return X_new
```


```python
X_rotated, min_pcs = get_min_pcs(X_scaled, 0.8)
X_new = project(X_rotated, min_pcs)
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_39_0.png)


## Clustering Model (K-Means)

Once we have reduced the dimensionality of the data, we can proceed with the clustering. The method for clustering chosen for this data is KMeans clustering. This works by assigning a random "mean point" in the data and adjusts each point by getting the closest points to the initally assigned mean point. The algorithm iterates this through multiple cycles, adjusting the mean point of each formed cluster until there are no more changes in the assigned mean point of each cluster. 

A prerequisite of the KMeans algorithm is that we will need to provide a number of clusters to use. In order to select the optimal number of clusters, we run an iteration of KMeans for k=2 until k=16 and plot the various internal validation criteria:
* SSE(Sum of Squares Error)
* Inter-Intra Cluster Range
* Calinski-Harabasz 
* Silhouette Coefficient


```python
kmeans_nba = KMeans(random_state=1337)
out = cluster_range(X_new, kmeans_nba, 16, actual=None)
```


```python
plot_internal(out['inertias'], out['chs'], out['iidrs'], out['scs']);
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_43_0.png)


The optimal number of k chosen is 4. This number is at the elbow point of the SSE and Inter-Intra cluster range, as well as retaining a high CH score, and Silhouette Coefficient. We proceed with clustering the data using KMeans with an optimal number of 4 clusters as its hyperparameter.


```python
# number of clusters
clusters = 4
y_predicted = out['ys'][clusters-2]

df4['y_predicted'] = y_predicted
```


```python
len_clusters = []
for n in set(y_predicted):
    c = df4.loc[y_predicted==n]
    cluster_count = len(c)
    len_clusters.append(cluster_count)
```


```python
fig, ax = plt.subplots()
ax.bar(Counter(y_predicted).keys(), Counter(y_predicted).values())
ax.set_ylabel('Number of Players')
ax.set_xlabel('Clusters')
ax.set_title('Number of Players per Cluster (k=4)');
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_47_0.png)



```python
X_players_new = TSNE(n_components=2,random_state=1337).fit_transform(X_new)
fig, ax = plt.subplots()
ax.scatter(X_players_new[:,0], X_players_new[:,1], c=list(y_predicted), 
           alpha=0.5)
ax.set_title('TSNE Projection of Clusters');
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_48_0.png)



```python
player_clusters = []
for i in range(clusters):
    grp = df4[df4.y_predicted==i]['Player'].to_list()
    player_clusters.append(dict(Counter(grp).most_common()))
```

## Analysis

Once we have clustered our player data, we first take a look at the player composition of each cluster. To get an overview of the players per cluster, we plot the names of the players in a word cloud. 


```python
fig, axs = plt.subplots(2,2, figsize=(16,9))
for i, ax in enumerate(fig.axes):
    freq = player_clusters[i]
    wordcloud_obj = WordCloud(background_color="white",
                              mask=None,
                              contour_width=1,
                              contour_color='white',
                              random_state=2018)
    wordcloud = wordcloud_obj.generate_from_frequencies(frequencies=freq)

    # Display the generated image
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_52_0.png)


From the cursory view given by the wordclouds, we can immediately see two distinct clusters: star players and centers. These are the clusters on the right hand side as denoted by the star cluster of Chris Paul, Dwyane Wade, Kevin Durant, and Lebron James, and the cluster of centers/bigs as denoted by LaMarcus Aldridge, Brook Lopez, Dwight Howard, Pau Gasol, and others. The other two clusters seem to be a mix of guards and forwards that are perhaps clustered based on their skill level as the cluster with Jared Dudley, J.J. Barea, C.J. Miles and others are known bench players or 6th man players for different teams. To get a clearer picture of the description per cluster, we can take a look at their average stats per player.


```python
df5 = df4.merge(df3[['Player', 'year', 'Age', 'Pos', 
                    'GS', 'MP', 'FG', 'TOV', 'PTS']], on=['Player', 'year'])
df5[df5.columns[-5:]] = df5[df5.columns[-5:]].astype(float)
df5.groupby('y_predicted')[df5.columns[2:]].mean().transpose()
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
      <th>y_predicted</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>G</th>
      <td>29.056604</td>
      <td>69.080184</td>
      <td>57.284817</td>
      <td>67.415716</td>
    </tr>
    <tr>
      <th>3P</th>
      <td>0.268257</td>
      <td>1.469401</td>
      <td>0.831792</td>
      <td>0.185044</td>
    </tr>
    <tr>
      <th>3P%</th>
      <td>0.216518</td>
      <td>0.355161</td>
      <td>0.348482</td>
      <td>0.163632</td>
    </tr>
    <tr>
      <th>2P</th>
      <td>0.947503</td>
      <td>4.079816</td>
      <td>1.734418</td>
      <td>4.375792</td>
    </tr>
    <tr>
      <th>2P%</th>
      <td>0.404438</td>
      <td>0.480169</td>
      <td>0.484890</td>
      <td>0.527292</td>
    </tr>
    <tr>
      <th>FT%</th>
      <td>0.670225</td>
      <td>0.800956</td>
      <td>0.767463</td>
      <td>0.693532</td>
    </tr>
    <tr>
      <th>eFG%</th>
      <td>0.402485</td>
      <td>0.502191</td>
      <td>0.507717</td>
      <td>0.522745</td>
    </tr>
    <tr>
      <th>ORB</th>
      <td>0.419645</td>
      <td>0.808756</td>
      <td>0.573916</td>
      <td>2.180608</td>
    </tr>
    <tr>
      <th>DRB</th>
      <td>1.235738</td>
      <td>3.544700</td>
      <td>2.069121</td>
      <td>4.950824</td>
    </tr>
    <tr>
      <th>AST</th>
      <td>0.950166</td>
      <td>4.185899</td>
      <td>1.461016</td>
      <td>1.582890</td>
    </tr>
    <tr>
      <th>STL</th>
      <td>0.351387</td>
      <td>1.171060</td>
      <td>0.551142</td>
      <td>0.730545</td>
    </tr>
    <tr>
      <th>BLK</th>
      <td>0.158713</td>
      <td>0.373548</td>
      <td>0.258048</td>
      <td>1.012928</td>
    </tr>
    <tr>
      <th>PF</th>
      <td>1.083685</td>
      <td>2.201290</td>
      <td>1.600514</td>
      <td>2.556907</td>
    </tr>
    <tr>
      <th>y_predicted</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>25.726970</td>
      <td>26.663594</td>
      <td>26.769406</td>
      <td>26.250951</td>
    </tr>
    <tr>
      <th>GS</th>
      <td>2.593785</td>
      <td>55.860829</td>
      <td>14.947489</td>
      <td>45.475285</td>
    </tr>
    <tr>
      <th>MP</th>
      <td>11.117203</td>
      <td>31.570968</td>
      <td>18.800400</td>
      <td>26.051838</td>
    </tr>
    <tr>
      <th>FG</th>
      <td>1.214650</td>
      <td>5.545438</td>
      <td>2.567009</td>
      <td>4.561090</td>
    </tr>
    <tr>
      <th>TOV</th>
      <td>0.619867</td>
      <td>2.091705</td>
      <td>0.899144</td>
      <td>1.483777</td>
    </tr>
    <tr>
      <th>PTS</th>
      <td>3.258713</td>
      <td>15.398894</td>
      <td>6.966781</td>
      <td>11.452091</td>
    </tr>
  </tbody>
</table>
</div>




```python
positions = []
for i in set(y_predicted):
    positions.append(dict(Counter(df5[df5['y_predicted'] == i]['Pos'])))
pd.DataFrame.from_records(positions).fillna(0)[['C', 'PF', 'SF', 'SG', 'PG']]
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
      <th>C</th>
      <th>PF</th>
      <th>SF</th>
      <th>SG</th>
      <th>PG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>78</td>
      <td>151</td>
      <td>189</td>
      <td>235</td>
      <td>226</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>111</td>
      <td>224</td>
      <td>305</td>
      <td>411</td>
    </tr>
    <tr>
      <th>2</th>
      <td>120</td>
      <td>302</td>
      <td>449</td>
      <td>517</td>
      <td>330</td>
    </tr>
    <tr>
      <th>3</th>
      <td>425</td>
      <td>303</td>
      <td>44</td>
      <td>7</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



By looking at the average stats per cluster, we are able to ascertain different insights from each cluster. As mentioned previously, there is the presence of the star cluster and the centers cluster. Based on these stats, we can describe each cluster as such:

* Cluster 0 - bench players. These are players with low overall stats, the most telling of which is the number of games played in the cluster, being just about half the number of games played compared to the other clusters. This also reflects in the amount of two pointers and three pointers that they've made as well as the comparatively low numbers they have in the other measures. This cluster may also be the rookies or new players in the league as they have the lowest average age of all clusters. 
* Cluster 1 - star players. These are the players who play the most minutes, indicating that they are the go-to guys of the team. This cluster also has the highest average stats in almost all categories, with a highlight being the number of games started, minute played, points, assists, and turnovers. The turnover rate in this cluster may be related to the number of minutes played, as well as the fact that star players usually have the ball in their hands the most while facing the stiffest defenses. The age of the players in this cluster is also telling as the average age is at around 26.66 years old, and the accepted "prime years" of an NBA player would coincide with this age range of about 26-30 years old. This cluster is predominantly made up of small forwards, shooting guards, and point guards, which coincide with the highest volume shooters in the league and make up the high scoring rate and shooting accuracy of this cluster.
* Cluster 2 - role players. These players form the role players in the team. While their numbers are better than those of cluster 0, there is a marked difference between them and the "elite" players in the league. Their numbers are generally lower than those of cluster 1 and cluster 3. Going back to the word cloud, we can see that the names of these players are composed mostly of bench players and starters who are role players, generally these are the players that you build around star players.
* Cluster 3 - big men. This cluster is predominantly composed of centers or big men as evidenced by the number of cneters and power forwards in the positions. As each team needs to start a center in the game, this role is usually very well defined in their stats such as high amount of rebounds and blocks, as well as a higher number of minutes played and highest personal fouls, as they are tasked with defending the rim from the driving opponents, making them more prone to fouling. As we saw in the feature vectors of the principal components above, this is a very well defined cluster as the feature vectors relating to big men/centers are all related or pointing in the same direction. As we expect from the PCA plot, this cluster is defined by the number of rebounds, personal fouls, and 2 pointers and 2 point accuracy.

Based on the clusters that were formed, the unsupervised KMeans algorithm seemed to cluster the players based on their skill set and skill level. Whereas we had truly elite guards/forwards in cluster 1, we also had the newbies or bench players in cluster 0. We also found that there is a concenctration of big men in cluster 3, possibly because of the strength of the feature vectors along this principal component. Thus, we deem that the clustering algorithm was able to cluster players based on both their skill set (usually related to their position or what they are able to bring to the game), and the level at which they execute at this skill set (given by the average stats for each cluster).

One of the insights we can garner from this analysis is that there is a chance that there are players that can be found in multiple clusters throughout their careers. This could be the player trajectory of starting out as a role player or bench player and eventually moving into the role of a star player during their prime years. Another possible career trajectory for this time period is that of a player coming down from their peak and being relegated into the role of a bench player or supporting player. In our clustering, the first scenario would indicate a move for players from cluster 2 into cluster 1, and the second scenario would involve a move from cluster 1 to cluster 2 or possibly cluster 0. In the past 10 years, there would be several players who fall into these categories: Klay Thompson was drafted in 2011 and played a minor role in the Golden State Warriors until their breakthrough season in 2014-2015 where he became a star player, Draymond Green similarly was drafted by GSW in 2012 and played a minor role up until the same season, and Kawhi Leonard was drafted in 2011 and played a minor role until the San Antonio Spurs won the championship in 2014 with which he won the Finals MVP award. We expect these players' career trajectories to place them in multiple clusters in our analysis. 


```python
players0 = set(df4[df4['y_predicted']==0]['Player'].to_list())
players1 = set(df4[df4['y_predicted']==1]['Player'].to_list())
players2 = set(df4[df4['y_predicted']==2]['Player'].to_list())
players3 = set(df4[df4['y_predicted']==3]['Player'].to_list())

player_analysis = ['Klay Thompson', 'Draymond Green', 'Kawhi Leonard']
compiler = []
for j in [players0, players1, players2, players3]:
    player_dict = dict(zip(player_analysis, [""] * 3))
    for i in player_analysis:
        player_dict[i] = i in j
    compiler.append(player_dict)
pd.DataFrame.from_records(compiler, columns=player_analysis)
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
      <th>Klay Thompson</th>
      <th>Draymond Green</th>
      <th>Kawhi Leonard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, we can see that all three of these players had different years in their career. Kawhi Leonard and Klay Thompson, both first round draft picks (15th and 11th, respectively) were most likely clustered first in cluster 2 as role players during the first half of their career, whereas Draymond Green, a second round draft pick, was first in the cluster 0 as a bench player. However, during their latter years, the three of them developed into full-fledged star players and being clustered into cluster 1 as star players for their team. We validate this by looking into the years in which each player were clustered into clusters 0, 1, and 2.


```python
for i in ['Klay Thompson', 'Kawhi Leonard']:
    print(f'{i}:')
    print(f'Cluster 2: {df4[(df4["y_predicted"]==2) & (df4["Player"]==i)]["Player_Unique"].to_list()}')
    print(f'Cluster 1: {df4[(df4["y_predicted"]==1) & (df4["Player"]==i)]["Player_Unique"].to_list()}')
    print('\n')

d = 'Draymond Green'
print(f'{d}:')
print(f'Cluster 0: {df4[(df4["y_predicted"]==0) & (df4["Player"]==d)]["Player_Unique"].to_list()}')
print(f'Cluster 1: {df4[(df4["y_predicted"]==1) & (df4["Player"]==d)]["Player_Unique"].to_list()}')
print('\n')
```

    Klay Thompson:
    Cluster 2: ['Klay Thompson_2012']
    Cluster 1: ['Klay Thompson_2013', 'Klay Thompson_2014', 'Klay Thompson_2015', 'Klay Thompson_2016', 'Klay Thompson_2017', 'Klay Thompson_2018', 'Klay Thompson_2019']
    
    
    Kawhi Leonard:
    Cluster 2: ['Kawhi Leonard_2012']
    Cluster 1: ['Kawhi Leonard_2013', 'Kawhi Leonard_2014', 'Kawhi Leonard_2015', 'Kawhi Leonard_2016', 'Kawhi Leonard_2017', 'Kawhi Leonard_2018', 'Kawhi Leonard_2019']
    
    
    Draymond Green:
    Cluster 0: ['Draymond Green_2013']
    Cluster 1: ['Draymond Green_2014', 'Draymond Green_2015', 'Draymond Green_2016', 'Draymond Green_2017', 'Draymond Green_2018', 'Draymond Green_2019']
    
    


From the list above, we can see that the career trajectories of Klay Thompson and Kawhi Leonard mirror each other down to the year that they entered and their rookie years serving as a role player before transitioning into elite status on their second year. Similarly, Draymond Green spent his first year as a bench player before breaking out as an elite star player in his second year.

# 2019 NBA Players

After clustering the NBA players for the 10 year period between 2009-2019, we take a look at the most recent NBA season and cluster the players for this year to validate whether or not these clusters have changed.


```python
df2019 = df[df['year'] == 2019]
df2019.drop_duplicates(['Player', 'year'], keep='first', inplace=True)
df2019 = df2019.reset_index()
df2019_players = df2019['Player']
df2019 = df2019[['Player', 'year', 'G', '3P', '3P%', '2P', '2P%', 'FT%',
                'eFG%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF']]
df2019 = df2019[~df2019['G'].str.contains('Did')]
df2019.dropna(how='any', inplace=True)
```


```python
df2019[df2019['Player'].str.contains('Ray')]
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
      <th>Player</th>
      <th>year</th>
      <th>G</th>
      <th>3P</th>
      <th>3P%</th>
      <th>2P</th>
      <th>2P%</th>
      <th>FT%</th>
      <th>eFG%</th>
      <th>ORB</th>
      <th>DRB</th>
      <th>AST</th>
      <th>STL</th>
      <th>BLK</th>
      <th>PF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>164</th>
      <td>Raymond Felton</td>
      <td>2019</td>
      <td>33.0</td>
      <td>0.6</td>
      <td>0.328</td>
      <td>1.1</td>
      <td>0.473</td>
      <td>0.923</td>
      <td>0.481</td>
      <td>0.1</td>
      <td>0.9</td>
      <td>1.6</td>
      <td>0.3</td>
      <td>0.2</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>458</th>
      <td>Ray Spalding</td>
      <td>2019</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.8</td>
      <td>0.568</td>
      <td>0.333</td>
      <td>0.532</td>
      <td>1.1</td>
      <td>2.4</td>
      <td>0.4</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>1.6</td>
    </tr>
  </tbody>
</table>
</div>



# PCA 2019


```python
X2019_scaled = scaler.fit_transform(df2019[df2019.columns[2:]])
```


```python
X2019_rotated, min_pcs2019 = get_min_pcs(X2019_scaled, 0.8)
X2019_new = project(X2019_rotated, min_pcs2019)
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_68_0.png)


In clustering the 2019 stats, we do dimensionality reduction through Principal Component Analysis in order to get the nuumber of PCs that will explain 80% of our total explained variance. This is similar to the one for the aggregated 10 year player cluster as we will be using 5 principal components.

# KMeans 2019


```python
kmeans_nba2019 = KMeans(random_state=1337)
out = cluster_range(X2019_new, kmeans_nba2019, 16, actual=None)
```


```python
plot_internal(out['inertias'], out['chs'], out['iidrs'], out['scs']);
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_72_0.png)



```python
# number of clusters
clusters = 4
y_predicted = out['ys'][clusters-2]

df2019['y_predicted'] = y_predicted
```


```python
len_clusters = []
for n in set(y_predicted):
    c = df2019.loc[y_predicted==n]
    cluster_count = len(c)
    len_clusters.append(cluster_count)
```


```python
len_clusters
```




    [146, 210, 56, 63]




```python
player_clusters = []
for i in range(clusters):
    grp = df2019[df2019.y_predicted==i]['Player'].to_list()
    player_clusters.append(dict(Counter(grp).most_common()))
```

# Analysis 2019


```python
fig, axs = plt.subplots(2,2, figsize=(16,9), dpi=300)
for i, ax in enumerate(fig.axes):
    freq = player_clusters[i]
    wordcloud_obj = WordCloud(background_color="white",
                              mask=None,
                              contour_width=1,
                              contour_color='white',
                              random_state=2018)
    wordcloud = wordcloud_obj.generate_from_frequencies(frequencies=freq)

    # Display the generated image
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_78_0.png)



```python
df2019[df2019.columns[2:-1]] = df2019[df2019.columns[2:-1]].astype(float)
grouped = df2019.merge(df3[['Player', 'year', 'Age', 'Pos', 
                    'GS', 'MP', 'FG', 'TOV', 'PTS']], on=['Player', 'year'])
grouped[grouped.columns[-5:]] = grouped[grouped.columns[-5:]].astype(float)
grouped.groupby('y_predicted')[grouped.columns[2:]].mean().transpose()
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
      <th>y_predicted</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>G</th>
      <td>67.527397</td>
      <td>48.119048</td>
      <td>66.785714</td>
      <td>20.619048</td>
    </tr>
    <tr>
      <th>3P</th>
      <td>1.734247</td>
      <td>0.736190</td>
      <td>0.532143</td>
      <td>0.341270</td>
    </tr>
    <tr>
      <th>3P%</th>
      <td>0.360445</td>
      <td>0.314329</td>
      <td>0.266179</td>
      <td>0.223016</td>
    </tr>
    <tr>
      <th>2P</th>
      <td>3.584247</td>
      <td>1.494286</td>
      <td>4.810714</td>
      <td>0.734921</td>
    </tr>
    <tr>
      <th>2P%</th>
      <td>0.494103</td>
      <td>0.520929</td>
      <td>0.575821</td>
      <td>0.407683</td>
    </tr>
    <tr>
      <th>FT%</th>
      <td>0.793096</td>
      <td>0.729205</td>
      <td>0.711804</td>
      <td>0.674730</td>
    </tr>
    <tr>
      <th>eFG%</th>
      <td>0.514411</td>
      <td>0.522633</td>
      <td>0.561929</td>
      <td>0.388841</td>
    </tr>
    <tr>
      <th>ORB</th>
      <td>0.789726</td>
      <td>0.616190</td>
      <td>2.312500</td>
      <td>0.296825</td>
    </tr>
    <tr>
      <th>DRB</th>
      <td>3.704110</td>
      <td>2.073333</td>
      <td>5.687500</td>
      <td>1.112698</td>
    </tr>
    <tr>
      <th>AST</th>
      <td>3.690411</td>
      <td>1.246667</td>
      <td>2.221429</td>
      <td>0.953968</td>
    </tr>
    <tr>
      <th>STL</th>
      <td>1.021233</td>
      <td>0.447143</td>
      <td>0.841071</td>
      <td>0.304762</td>
    </tr>
    <tr>
      <th>BLK</th>
      <td>0.395205</td>
      <td>0.281429</td>
      <td>1.101786</td>
      <td>0.112698</td>
    </tr>
    <tr>
      <th>PF</th>
      <td>2.203425</td>
      <td>1.547143</td>
      <td>2.701786</td>
      <td>0.936508</td>
    </tr>
    <tr>
      <th>y_predicted</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>26.493151</td>
      <td>26.214286</td>
      <td>25.392857</td>
      <td>24.603175</td>
    </tr>
    <tr>
      <th>GS</th>
      <td>50.020548</td>
      <td>8.004762</td>
      <td>49.125000</td>
      <td>1.301587</td>
    </tr>
    <tr>
      <th>MP</th>
      <td>29.271918</td>
      <td>15.933810</td>
      <td>26.544643</td>
      <td>9.588889</td>
    </tr>
    <tr>
      <th>FG</th>
      <td>5.315753</td>
      <td>2.232381</td>
      <td>5.341071</td>
      <td>1.073016</td>
    </tr>
    <tr>
      <th>TOV</th>
      <td>1.806849</td>
      <td>0.721905</td>
      <td>1.608929</td>
      <td>0.509524</td>
    </tr>
    <tr>
      <th>PTS</th>
      <td>14.733562</td>
      <td>6.013810</td>
      <td>13.685714</td>
      <td>2.942857</td>
    </tr>
  </tbody>
</table>
</div>



When we cluster all the players for just one year, we can see that there are roughly the same clusters as in the aggregate clustering. There are still clusters for the following:

* Big men/Defensive - Cluster 2. These are players that lead all clusters in rebounding and shot blocking as was in the aggregate clustering.
* Elite/Stars - Cluster 0. These are the star players who play the most minutes and score the most points.
* Bench Players - Cluster 3. These are the players with the lowest games played and overall stats. Similar to the aggregate clusters, these are developing players or rookies, as evident with their low average age.
* Role Players - Cluster 1. These players are the role players for every team. Similar again to the aggregate clustering, we see that they are relatively balanced in their stats and can contribute in many ways.

# Clustering 3 Point Shooters

As we've seen in the EDA section, there has been a growing trend of 3 point shooting in the league, led by the teams of Steph Curry and James Harden. In this section, we look to cluster the 3 point shooters in the league to see what differentiates them from one another. In the selection of players to cluster, we limited these to the players who have attempted at least 200 3 point shots throughout the course of the season, which is the basis for candidacy for the 3 point shooting crown of the NBA. We are left with 158 players after filtering for this criteria.


```python
df6 = pd.read_sql('''SELECT * FROM shot_finder''', conn)
df6[df6.columns[4:]] = df6[df6.columns[4:]].astype(float)
```


```python
df6 = df6[df6['3PA'] > 200]
df7 = df6[['3PA', '3P%', "%Ast'd"]]
```

To be able to filter out the elite 3 point shooters in the league, we will be looking at three factors: the volume of shots, accuracy of their shots, and the percentage of shots they can create on their own. These stats correspond to `3PA`, `3P%`, and `%Ast'd`, respectively and reflect the three biggest factors that are relevant in the NBA today: the number, accuracy, and skill in creating and making 3 point shots. At the end of this clustering analysis, we will be able to bucket the different kind of three point shooters in the league.

As there are only 3 features to be used, we will forego the Principal Component Analysis section and go straight to clustering these players.


```python
kmeans_3p = KMeans(random_state=1337)
out = cluster_range(df7.to_numpy(), kmeans_3p, 16, actual=None)
```


```python
plot_internal(out['inertias'], out['chs'], out['iidrs'], out['scs']);
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_87_0.png)


Based on the internal validation criteria above, we select 6 as the optimal number of clusters. This is chosen as the elbow point of the inter-intra cluster range and SSE, as well as maintaining a high silhouette score (although the difference between the max and min Silhouette score is only ~0.5).


```python
# number of clusters
clusters3p = 6
y_predicted3p = out['ys'][clusters3p-2]

df6['y_predicted'] = y_predicted3p
```


```python
print('Average Player Stats per Cluster:')
df6.groupby('y_predicted')[['3PA', '3P%', "%Ast'd"]].mean().transpose()
```

    Average Player Stats per Cluster:





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
      <th>y_predicted</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3PA</th>
      <td>653.909091</td>
      <td>378.472222</td>
      <td>229.42000</td>
      <td>465.083333</td>
      <td>1028.000</td>
      <td>305.25000</td>
    </tr>
    <tr>
      <th>3P%</th>
      <td>0.385182</td>
      <td>0.371667</td>
      <td>0.35136</td>
      <td>0.358000</td>
      <td>0.368</td>
      <td>0.36225</td>
    </tr>
    <tr>
      <th>%Ast'd</th>
      <td>0.714091</td>
      <td>0.818750</td>
      <td>0.87790</td>
      <td>0.769167</td>
      <td>0.161</td>
      <td>0.84625</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Count of players per cluster:')
df6.groupby('y_predicted')[['3PA']].count().transpose()
```

    Count of players per cluster:





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
      <th>y_predicted</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3PA</th>
      <td>11</td>
      <td>36</td>
      <td>50</td>
      <td>24</td>
      <td>1</td>
      <td>36</td>
    </tr>
  </tbody>
</table>
</div>



The summary of each stat above shows the stark difference in the different clusters that were formed, with their being an outlier cluster in cluster 4. When we take a look at the number of players in each cluster, there is a clear outlier in the players this season with only one player belonging to cluster 4.

Based on the stats per cluster, we can see:
* Cluster 2 - spot up shooters. These are the shooters who prefer staying on the perimiter and wait for the slashers or ball handlers to pass them the ball when they are open. This is evident in the low number of attempts, and the high assist rate for their three pointers.
* Cluster 5 - mix of spot up and low volume shooters. These are players who are not traditionally purely three point shooters but can knock down the shot when called upon. Additionally, they have a high assist rate meaning that they are more likely also camped out in the perimeter but do venture out on their own.
* Cluster 1 - spot up sharpshooters. These players are mostly assisted on their 3 point makes, but also shoot them at a high accuracy. As shown by their stats, they do not tend to shoot a lot of 3 pointers but are confident when they do.

For clusters 0, 3, and 4, we look into the breakdown of players per cluster to get a better understanding of the composition of each.


```python
rel_cols = ['Player', 'Tm', 'G', '3PA', '3P%', "%Ast'd"]
```


```python
df6[df6['y_predicted']==3].sort_values('3PA', ascending=False)[rel_cols].head(10)
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
      <th>Player</th>
      <th>Tm</th>
      <th>G</th>
      <th>3PA</th>
      <th>3P%</th>
      <th>%Ast'd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>426</th>
      <td>Blake Griffin</td>
      <td>DET</td>
      <td>75.0</td>
      <td>522.0</td>
      <td>0.362</td>
      <td>0.561</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Jae Crowder</td>
      <td>UTA</td>
      <td>80.0</td>
      <td>522.0</td>
      <td>0.331</td>
      <td>0.942</td>
    </tr>
    <tr>
      <th>423</th>
      <td>Donovan Mitchell</td>
      <td>UTA</td>
      <td>77.0</td>
      <td>519.0</td>
      <td>0.362</td>
      <td>0.580</td>
    </tr>
    <tr>
      <th>443</th>
      <td>Luka Dončić</td>
      <td>DAL</td>
      <td>72.0</td>
      <td>514.0</td>
      <td>0.327</td>
      <td>0.423</td>
    </tr>
    <tr>
      <th>194</th>
      <td>Brook Lopez</td>
      <td>MIL</td>
      <td>81.0</td>
      <td>512.0</td>
      <td>0.365</td>
      <td>0.952</td>
    </tr>
    <tr>
      <th>330</th>
      <td>Joe Ingles</td>
      <td>UTA</td>
      <td>82.0</td>
      <td>483.0</td>
      <td>0.391</td>
      <td>0.825</td>
    </tr>
    <tr>
      <th>444</th>
      <td>Trae Young</td>
      <td>ATL</td>
      <td>80.0</td>
      <td>482.0</td>
      <td>0.324</td>
      <td>0.423</td>
    </tr>
    <tr>
      <th>374</th>
      <td>Tim Hardaway</td>
      <td>TOT</td>
      <td>65.0</td>
      <td>477.0</td>
      <td>0.340</td>
      <td>0.728</td>
    </tr>
    <tr>
      <th>412</th>
      <td>Khris Middleton</td>
      <td>MIL</td>
      <td>77.0</td>
      <td>474.0</td>
      <td>0.378</td>
      <td>0.603</td>
    </tr>
    <tr>
      <th>344</th>
      <td>Reggie Jackson</td>
      <td>DET</td>
      <td>82.0</td>
      <td>471.0</td>
      <td>0.369</td>
      <td>0.793</td>
    </tr>
  </tbody>
</table>
</div>



For cluster3, we can see that these players are elite players who are able to create their own shot and this cluster is defined by 3 point shooters who are able to dribble and spot up and make their 3 pointers with high accuracy. In this cluster, we can see the great 3 point shooters in the league with the likes of Trae Young, Khris Middleton, and others.


```python
df6[df6['y_predicted']==0].sort_values('3PA', ascending=False)[rel_cols].head()
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
      <th>Player</th>
      <th>Tm</th>
      <th>G</th>
      <th>3PA</th>
      <th>3P%</th>
      <th>%Ast'd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>383</th>
      <td>Stephen Curry</td>
      <td>GSW</td>
      <td>69.0</td>
      <td>810.0</td>
      <td>0.437</td>
      <td>0.689</td>
    </tr>
    <tr>
      <th>392</th>
      <td>Paul George</td>
      <td>OKC</td>
      <td>77.0</td>
      <td>757.0</td>
      <td>0.386</td>
      <td>0.671</td>
    </tr>
    <tr>
      <th>442</th>
      <td>Kemba Walker</td>
      <td>CHO</td>
      <td>82.0</td>
      <td>731.0</td>
      <td>0.356</td>
      <td>0.438</td>
    </tr>
    <tr>
      <th>319</th>
      <td>Buddy Hield</td>
      <td>SAC</td>
      <td>82.0</td>
      <td>651.0</td>
      <td>0.427</td>
      <td>0.842</td>
    </tr>
    <tr>
      <th>439</th>
      <td>Damian Lillard</td>
      <td>POR</td>
      <td>80.0</td>
      <td>643.0</td>
      <td>0.369</td>
      <td>0.460</td>
    </tr>
  </tbody>
</table>
</div>



In cluster0, we see the truly elite 3 point shooters in the league with star players such as Stephen Curry, Paul George, Kemba Walker, and Damian Lillard. These players are able to create their own shot and shoot at a very high volume while maintaining their accuracy. A point to note here is that Steph Curry is by far the best shooter of this cluster as he shoots more three pointers and has the highest accuracy, while still being able to create his own shots.


```python
df6[df6['y_predicted']==4].sort_values('3PA', ascending=False).head()
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
      <th>Rk</th>
      <th>Player</th>
      <th>Season</th>
      <th>Tm</th>
      <th>G</th>
      <th>FG</th>
      <th>FGA</th>
      <th>FG%</th>
      <th>FGX</th>
      <th>3P</th>
      <th>3PA</th>
      <th>3P%</th>
      <th>3PX</th>
      <th>eFG%</th>
      <th>Ast'd</th>
      <th>%Ast'd</th>
      <th>y_predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>449</th>
      <td>450</td>
      <td>James Harden</td>
      <td>2018-19</td>
      <td>HOU</td>
      <td>78.0</td>
      <td>378.0</td>
      <td>1028.0</td>
      <td>0.368</td>
      <td>650.0</td>
      <td>378.0</td>
      <td>1028.0</td>
      <td>0.368</td>
      <td>650.0</td>
      <td>0.552</td>
      <td>61.0</td>
      <td>0.161</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



In cluster4, we find the true outlier of the 2018-19 NBA season in 3 pointers. James Harden not only shot the most 3 point attempts but also managed to shoot at a relatively high 36.8%, while also having the lowest assisted shot percentage which shows that he is taking most of these 3 point shots off his own dribbles. This also most likely means that these shots are contested as he would be holding the ball at the time before his shot. In this cluster, he is the only player as he is by far the player who shot the most 3's and has the lowest assisted 3 point rate.


```python
X_players_new3p = TSNE(n_components=2,random_state=1337).fit_transform(df7.to_numpy())
fig, ax = plt.subplots()
ax.scatter(X_players_new3p[:,0], X_players_new3p[:,1], c=list(y_predicted3p), 
           alpha=0.5)
ax.set_title('TSNE Projection of Clusters');
```


![png](/images/nba-analysis/Lab_Lab%205_NBA%20Clustering_100_0.png)


Lastly, when we plot the clusters on the tSNE representation of the dataset, we see that there is a very good separation between the clusters as there are no overlaps that can be seen. In the lower left quadrant, we see the very elite shooters, with one outlier (James Harden). In testing out the different values for k, it showed that even at different values of k, James Harden still clusters on his own as his stats are above and beyond the others for this season.

# Appendix

## Players per 3 Point Cluster (clusters 2, 3, 5)

Cluster 2


```python
df6[df6['y_predicted']==2].sort_values('3PA', ascending=False).head()
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
      <th>Rk</th>
      <th>Player</th>
      <th>Season</th>
      <th>Tm</th>
      <th>G</th>
      <th>FG</th>
      <th>FGA</th>
      <th>FG%</th>
      <th>FGX</th>
      <th>3P</th>
      <th>3PA</th>
      <th>3P%</th>
      <th>3PX</th>
      <th>eFG%</th>
      <th>Ast'd</th>
      <th>%Ast'd</th>
      <th>y_predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>144</th>
      <td>145</td>
      <td>Jonathan Isaac</td>
      <td>2018-19</td>
      <td>ORL</td>
      <td>72.0</td>
      <td>86.0</td>
      <td>266.0</td>
      <td>0.323</td>
      <td>180.0</td>
      <td>86.0</td>
      <td>266.0</td>
      <td>0.323</td>
      <td>180.0</td>
      <td>0.485</td>
      <td>85.0</td>
      <td>0.988</td>
      <td>2</td>
    </tr>
    <tr>
      <th>224</th>
      <td>225</td>
      <td>Garrett Temple</td>
      <td>2018-19</td>
      <td>TOT</td>
      <td>72.0</td>
      <td>90.0</td>
      <td>264.0</td>
      <td>0.341</td>
      <td>174.0</td>
      <td>90.0</td>
      <td>264.0</td>
      <td>0.341</td>
      <td>174.0</td>
      <td>0.511</td>
      <td>84.0</td>
      <td>0.933</td>
      <td>2</td>
    </tr>
    <tr>
      <th>236</th>
      <td>237</td>
      <td>Joel Embiid</td>
      <td>2018-19</td>
      <td>PHI</td>
      <td>61.0</td>
      <td>79.0</td>
      <td>263.0</td>
      <td>0.300</td>
      <td>184.0</td>
      <td>79.0</td>
      <td>263.0</td>
      <td>0.300</td>
      <td>184.0</td>
      <td>0.451</td>
      <td>73.0</td>
      <td>0.924</td>
      <td>2</td>
    </tr>
    <tr>
      <th>428</th>
      <td>429</td>
      <td>Dwyane Wade</td>
      <td>2018-19</td>
      <td>MIA</td>
      <td>68.0</td>
      <td>86.0</td>
      <td>261.0</td>
      <td>0.330</td>
      <td>175.0</td>
      <td>86.0</td>
      <td>261.0</td>
      <td>0.330</td>
      <td>175.0</td>
      <td>0.494</td>
      <td>47.0</td>
      <td>0.547</td>
      <td>2</td>
    </tr>
    <tr>
      <th>190</th>
      <td>191</td>
      <td>Tyler Johnson</td>
      <td>2018-19</td>
      <td>TOT</td>
      <td>56.0</td>
      <td>90.0</td>
      <td>260.0</td>
      <td>0.346</td>
      <td>170.0</td>
      <td>90.0</td>
      <td>260.0</td>
      <td>0.346</td>
      <td>170.0</td>
      <td>0.519</td>
      <td>86.0</td>
      <td>0.956</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 3


```python
df6[df6['y_predicted']==3].sort_values('3PA', ascending=False).head()
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
      <th>Rk</th>
      <th>Player</th>
      <th>Season</th>
      <th>Tm</th>
      <th>G</th>
      <th>FG</th>
      <th>FGA</th>
      <th>FG%</th>
      <th>FGX</th>
      <th>3P</th>
      <th>3PA</th>
      <th>3P%</th>
      <th>3PX</th>
      <th>eFG%</th>
      <th>Ast'd</th>
      <th>%Ast'd</th>
      <th>y_predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>426</th>
      <td>427</td>
      <td>Blake Griffin</td>
      <td>2018-19</td>
      <td>DET</td>
      <td>75.0</td>
      <td>189.0</td>
      <td>522.0</td>
      <td>0.362</td>
      <td>333.0</td>
      <td>189.0</td>
      <td>522.0</td>
      <td>0.362</td>
      <td>333.0</td>
      <td>0.543</td>
      <td>106.0</td>
      <td>0.561</td>
      <td>3</td>
    </tr>
    <tr>
      <th>206</th>
      <td>207</td>
      <td>Jae Crowder</td>
      <td>2018-19</td>
      <td>UTA</td>
      <td>80.0</td>
      <td>173.0</td>
      <td>522.0</td>
      <td>0.331</td>
      <td>349.0</td>
      <td>173.0</td>
      <td>522.0</td>
      <td>0.331</td>
      <td>349.0</td>
      <td>0.497</td>
      <td>163.0</td>
      <td>0.942</td>
      <td>3</td>
    </tr>
    <tr>
      <th>423</th>
      <td>424</td>
      <td>Donovan Mitchell</td>
      <td>2018-19</td>
      <td>UTA</td>
      <td>77.0</td>
      <td>188.0</td>
      <td>519.0</td>
      <td>0.362</td>
      <td>331.0</td>
      <td>188.0</td>
      <td>519.0</td>
      <td>0.362</td>
      <td>331.0</td>
      <td>0.543</td>
      <td>109.0</td>
      <td>0.580</td>
      <td>3</td>
    </tr>
    <tr>
      <th>443</th>
      <td>444</td>
      <td>Luka Dončić</td>
      <td>2018-19</td>
      <td>DAL</td>
      <td>72.0</td>
      <td>168.0</td>
      <td>514.0</td>
      <td>0.327</td>
      <td>346.0</td>
      <td>168.0</td>
      <td>514.0</td>
      <td>0.327</td>
      <td>346.0</td>
      <td>0.490</td>
      <td>71.0</td>
      <td>0.423</td>
      <td>3</td>
    </tr>
    <tr>
      <th>194</th>
      <td>195</td>
      <td>Brook Lopez</td>
      <td>2018-19</td>
      <td>MIL</td>
      <td>81.0</td>
      <td>187.0</td>
      <td>512.0</td>
      <td>0.365</td>
      <td>325.0</td>
      <td>187.0</td>
      <td>512.0</td>
      <td>0.365</td>
      <td>325.0</td>
      <td>0.548</td>
      <td>178.0</td>
      <td>0.952</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 5


```python
df6[df6['y_predicted']==5].sort_values('3PA', ascending=False).head()
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
      <th>Rk</th>
      <th>Player</th>
      <th>Season</th>
      <th>Tm</th>
      <th>G</th>
      <th>FG</th>
      <th>FGA</th>
      <th>FG%</th>
      <th>FGX</th>
      <th>3P</th>
      <th>3PA</th>
      <th>3P%</th>
      <th>3PX</th>
      <th>eFG%</th>
      <th>Ast'd</th>
      <th>%Ast'd</th>
      <th>y_predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>146</th>
      <td>147</td>
      <td>Dāvis Bertāns</td>
      <td>2018-19</td>
      <td>SAS</td>
      <td>76.0</td>
      <td>145.0</td>
      <td>338.0</td>
      <td>0.429</td>
      <td>193.0</td>
      <td>145.0</td>
      <td>338.0</td>
      <td>0.429</td>
      <td>193.0</td>
      <td>0.643</td>
      <td>143.0</td>
      <td>0.986</td>
      <td>5</td>
    </tr>
    <tr>
      <th>296</th>
      <td>297</td>
      <td>Kelly Oubre</td>
      <td>2018-19</td>
      <td>TOT</td>
      <td>69.0</td>
      <td>108.0</td>
      <td>338.0</td>
      <td>0.320</td>
      <td>230.0</td>
      <td>108.0</td>
      <td>338.0</td>
      <td>0.320</td>
      <td>230.0</td>
      <td>0.479</td>
      <td>94.0</td>
      <td>0.870</td>
      <td>5</td>
    </tr>
    <tr>
      <th>369</th>
      <td>370</td>
      <td>Terry Rozier</td>
      <td>2018-19</td>
      <td>BOS</td>
      <td>78.0</td>
      <td>119.0</td>
      <td>337.0</td>
      <td>0.353</td>
      <td>218.0</td>
      <td>119.0</td>
      <td>337.0</td>
      <td>0.353</td>
      <td>218.0</td>
      <td>0.530</td>
      <td>88.0</td>
      <td>0.739</td>
      <td>5</td>
    </tr>
    <tr>
      <th>163</th>
      <td>164</td>
      <td>Lauri Markkanen</td>
      <td>2018-19</td>
      <td>CHI</td>
      <td>52.0</td>
      <td>120.0</td>
      <td>332.0</td>
      <td>0.361</td>
      <td>212.0</td>
      <td>120.0</td>
      <td>332.0</td>
      <td>0.361</td>
      <td>212.0</td>
      <td>0.542</td>
      <td>117.0</td>
      <td>0.975</td>
      <td>5</td>
    </tr>
    <tr>
      <th>272</th>
      <td>273</td>
      <td>Malik Monk</td>
      <td>2018-19</td>
      <td>CHO</td>
      <td>71.0</td>
      <td>109.0</td>
      <td>330.0</td>
      <td>0.330</td>
      <td>221.0</td>
      <td>109.0</td>
      <td>330.0</td>
      <td>0.330</td>
      <td>221.0</td>
      <td>0.495</td>
      <td>98.0</td>
      <td>0.899</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
