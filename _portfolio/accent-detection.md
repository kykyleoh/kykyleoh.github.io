---
title: "Say What? Accent Detection using Machine Learning"
excerpt: "Application of frequency transformations and machine learning classifiers for english foreign and native accent prediction.<br/><img src='/images/accent-detection/accent-detection-titlecard.png'>"
collection: portfolio
---

<h1>Overview</h1>
<p>This project was done as a final individual requirement for Machine Learning class in AIM Msc Data Science. In this project, I apply multiple transformations to sound data and convert this to data that is readable by traditional machine learning tools.</p>

<h2>Note:</h2>
<p>Most of the following is written as a technical report, and as such contains a significant amount of code. If you're interested in the methodoloy and results, I highly recommend browsing through the presentation deck instead.</p>

The presentation can be found here: [pdf](/files/Say-What-accent-detection.pdf)

# Executive Summary:

Accent detection could be very beneficial for voice processing technologies. As different big companies invest more into the voice AI assistant industry, detecting a users accent could yield more accurate results for the algorithms as well as unlock more information on the user. This study uses XGBoostClassifier to achieve a 59.4% accuracy on detecting the accent of three different native and non-native english speakers. These participants in the dataset are either American, Korean, or Chinese nationals that spoke the same english phrase. The analysis shows that the classifier was able to achieve a relatively high recall and precision on determining Korean speakers of english as opposed to those of Chinese, which had a low precision and recall score.


```python
conf_mat = pd.DataFrame(np.array([[352,  73, 170], [113, 223, 161], [120,  86, 484]]))
conf_mat.columns = ['English', 'Chinese', 'Korean']
conf_mat.index = ['English', 'Chinese', 'Korean']
conf_mat
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
      <th>English</th>
      <th>Chinese</th>
      <th>Korean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>English</th>
      <td>352</td>
      <td>73</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Chinese</th>
      <td>113</td>
      <td>223</td>
      <td>161</td>
    </tr>
    <tr>
      <th>Korean</th>
      <td>120</td>
      <td>86</td>
      <td>484</td>
    </tr>
  </tbody>
</table>
</div>



                      precision    recall  f1-score   support

           english       0.60      0.59      0.60       595
           chinese       0.58      0.45      0.51       497
           korean        0.59      0.70      0.64       690

This could potentially be due to the close difference in the speaking style of the two languages, which may lead to the classifier to have a hard time distinguishing Chinese speakers of english as opposed to those of Korean speakers who may have a more distinct way of speaking in english. This may also be due to the pre-processing of the signals as there could be significant levels of noise in the data taht could skew the results. Aside from this, the signals were converted to MFCCs (Mel Frequency Cepstral Coefficients) that are more reminiscent of the way human naturally hear different frequencies (log-scaled). A different approach could also be utilized in the future using Neural Networks and converting the signals into images that could be processed as the image of the overall speech pattern of each individual.

# The Dataset: 
## Wildcat Corpus of Native- and Foreign-Accented English
* From Northwestern University
* Accessed on August 5, 2019 from: http://groups.linguistics.northwestern.edu/speech_comm_group/wildcat/
* 86 speakers of different Nationalities saying a common phrase
* Phrase: "Please call Stella.  Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."


# Load Prerequisites


```python
import pandas as pd
from collections import Counter
import librosa
import matplotlib.pyplot as plt
%matplotlib inline
import librosa.display as _display
import numpy as np
import glob
from tqdm.autonotebook import tqdm
pd.options.display.float_format = '{:,.5g}'.format
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.fftpack import fft, ifft
import machumachine
from itertools import compress
from sklearn.model_selection import GridSearchCV
```

# Load Prerequisite Functions For Signal Selection

* `findsilence` - finds the "silent" sections of the signal

* `merger` - merges the silent sections to the original sample signal

* `shade_silence` - returns a plot of the signal with the "silent" areas shaded. Also returns the index (start,stop) of each non-silent area in the signal

## Load all signal files


```python
def findsilence(y,sr,ind_i):
    hop = int(round(sr*0.18)) #hop and width defines search window
    width = int(sr*0.18)
    n_slice = int(len(y)/hop)
    starts = np.arange(n_slice)*hop
    ends = starts+width
    if hop != width:
        cutoff = np.argmax(ends>len(y))
        starts = starts[:cutoff]
        ends = ends[:cutoff]
        n_slice = len(starts)
    mask = [i for i in map(lambda i: np.dot(y[starts[i]:ends[i]],y[starts[i]:ends[i]])/width < (0.5 * np.dot(y,y)/len(y)), range(n_slice))]
    starts =  list(compress(starts+ind_i,mask))
    ends = list(compress(ends+ind_i,mask))
    return zip(starts,ends)

def merger(tulist):
    tu=[]
    for tt in tulist:
        tu.append(tt)
    tu = tuple(tu)
    cnt = Counter(tu)
    res = [i for i in filter(lambda x: cnt[x]<2, tu)]
#     return res
#     return [i for i in map(lambda x: tuple(x),np.array(res).reshape((19,2)))]
    return [i for i in map(lambda x: tuple(x),np.array(res).reshape(int(len(res)),2))]

def shade_silence(filename,start=0,end=None,disp=True,output=False, itr='', save=None):
    """Find signal (as output) or silence (as shaded reagion  in plot) in a audio file
    filename: (filepath) works best with .wav format
    start/end: (float or int) start/end time for duration of interest in second (default= entire length)
    disp: (bool) whether to display a plot(default= True)
    output: (bool) whether to return an output (default = False)
    itr: (int) iteration use for debugging purpose
    save: (str) filename to save to
    """
    try:
        y, sr = librosa.load(filename)
    except:
        obj = thinkdsp.read_wave(filename)
        y = obj.ys
        sr = obj.framerate
        print(itr, ' : librosa.load failed for '+filename)

    t = np.arange(len(y))/sr

    i = int(round(start * sr))
    if end != None:
        j = int(round(end * sr))
    else:
        j = len(y)
    fills = findsilence(y[i:j],sr,i)
    if disp:
        fig, ax = plt.subplots(dpi=200, figsize=(15,8))
        ax.set_title(filename)
        ax.plot(t[i:j],y[i:j])
        ax.set_xlabel('Time (s)')
        
    if fills != None:
        shades = [i for i in map(lambda x: (max(x[0],i),min(x[1],j)), fills)]
        if len(shades)>0:
            shades = merger(shades)
            if disp:
                for s in shades:
                    ax.axvspan(s[0]/sr, s[1]/sr, alpha=0.5, color='r')
    
    if save:
        fig.savefig(save)
        
    if len(shades)>1:
        live = map(lambda i: (shades[i][1],shades[i+1][0]), range(len(shades)-1))
    elif len(shades)==1:
        a = [i,shades[0][0],shades[0][1],j]
        live = filter(lambda x: x != None, map(lambda x: tuple(x) if x[0]!=x[1] else None,np.sort(a).reshape((int(len(a)/2),2))))
    else:
        live = [(i,j)]
    if output:
        return [i for i in live], sr, len(y)
```

# Load and Slice each audio by time to generate new data points


```python
# generates new y values that are cleaned

new_ys = []

for i in tqdm(final['filepaths']):
    
    new_y = []
    y, sr = librosa.load(i)
    
    silences, b, c = shade_silence(i, output=True, disp=False)
    
    for j in range(len(silences)):
        if silences[j][0] != silences[j][1]:
            new_y.extend(y[silences[j][0]:silences[j][1]])
    
    new_ys.append(new_y)
```


    HBox(children=(IntProgress(value=0, max=72), HTML(value='')))


    


## Create a function `splitter` to aggregate the signal files
* `splitter` - creates a new signal, compressing the original signal by removing the silent areas and leaving only those timeframes with sound


```python
def splitter(y_list, target, interval=1, original_sr=22050, n_mfcc=20):
    
    '''
    Converts a list of audio signals to mfcc bands and slices and resamples the audio signals based on 
    original_sr and interval. 
    
    Returns a 2D n x m array. n is the number of data points, with m-1 features as each mfcc band value for
    the interval or time step selected.
    '''
    
    import math
    
    # check for the shortest sample

    newys_lengths = []

    for i in y_list:
        newys_lengths.append(len(i))
        
    max_length = int(np.array(newys_lengths).min()/22050)
    
    # create bins that match the interval length and sampling rate
    sec_bins = []
    
    steps = np.arange(0, (max_length*original_sr), (original_sr*interval))

    for i in range(len(steps)-1):
        sec_bins.append([steps[i],steps[i+1]])
    
    # create new slices of each audio signal in y_list
    print(f'Length of time bins = {len(sec_bins)}')
    print('Generating slices of X.')
    
    new_X = []

    for i in tqdm(sec_bins):

        for audio in y_list:

            new_X.append(np.array(audio[int(i[0]):int(i[1])]))

    new_X = np.array(new_X)
    
    print('Checking for empty arrays.')
    
    counter = 0
    
    shapes = [i.shape for i in new_X]
    clean_index = []
    
    for i in range(len(shapes)):
        if shapes[i] == (0,):
            counter += 1
        else:
            clean_index.append(i)
            
    proceed = 'y'
    
    if counter > 0:
        proceed = eval(input(f'Number of empty arrays is {counter}. Proceed and delete these arrays? (y/n?)'))
    else:
        print('No empty arrays.\n')
    
    if proceed == 'y':

        new_X = new_X[clean_index]
        
    print('Slicing original audio files.')
    
    new_Xs = []

    for i in tqdm(range(len(new_X))):

        new_Xs.append(librosa.feature.mfcc(new_X[i], sr=original_sr, n_mfcc=n_mfcc))

    new_Xs = np.array(new_Xs)

    try:
        new_Xs = new_Xs.reshape(len(new_X), (int(math.ceil((original_sr*interval)/512)) * n_mfcc))
    except:
        print(new_Xs.shape)
        print(len(new_Xs))
        
    # generate new targets    
    try:
        new_target = np.array(list(target.to_numpy()) * len(sec_bins))
    except AttributeError:
        new_target = np.array(list(target) * len(sec_bins))
    
    return new_Xs, new_target
```


```python
X, target = splitter(new_ys, target, interval=0.1, n_mfcc=40)
```

    Length of time bins = 99
    Generating slices of X.



    HBox(children=(IntProgress(value=0, max=99), HTML(value='')))


    
    Checking for empty arrays.
    No empty arrays
    Slicing original audio files.



    HBox(children=(IntProgress(value=0, max=7128), HTML(value='')))


    



```python
X.shape
```




    (7128, 200)



# Classification Models

## Load train_test_split


```python
X_train, X_test, y_train, y_test = train_test_split(X, target, random_state=1028)
```

## Random Forest Classifier


```python
param_grids = {'max_depth': [3, 4, 6],
              'min_samples_leaf': [2, 3, 4],
              'max_features': [3, 2, 5],
            'n_estimators':[200,400,600]
}   

est2 = RandomForestClassifier()
gs_cv2 = GridSearchCV(est, param_grids, n_jobs=-1, verbose=10, cv=3).fit(X_train, y_train)
print(gs_cv2.best_params_)
```

    Fitting 3 folds for each of 81 candidates, totalling 243 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    1.2s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:    3.6s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:    4.8s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:    7.9s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   10.4s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:   12.8s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:   16.0s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:   21.6s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:   26.7s
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:   30.8s
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:   35.4s
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed:   40.1s
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:   48.8s
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed:   58.8s
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  1.1min
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=-1)]: Done 243 out of 243 | elapsed:  1.6min finished


    {'max_depth': 6, 'max_features': 5, 'min_samples_leaf': 2, 'n_estimators': 600}



```python
gs_cv2.score(X_test, y_test)
```




    0.4618406285072952




```python
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
```


```python
new_y_pred2 = gs_cv.predict(X_test)
```


```python
confusion_matrix(y_test, new_y_pred2)
```




    array([[370,  27, 173],
           [ 97, 198, 219],
           [ 74,  15, 609]])




```python
accuracy_score(y_test, new_y_pred)
```




    0.6604938271604939




```python
print(classification_report(y_test, new_y_pred))
```

                  precision    recall  f1-score   support
    
               1       0.68      0.65      0.67       570
               2       0.82      0.39      0.53       514
               3       0.61      0.87      0.72       698
    
       micro avg       0.66      0.66      0.66      1782
       macro avg       0.71      0.64      0.64      1782
    weighted avg       0.70      0.66      0.65      1782
    


## Gradient Boosting Classifier


```python
param_grids = {'learning_rate': [.2, 0.1, 0.05, 0.02, 0.01],
              'max_depth': [3, 4, 6, 8],
              'min_samples_leaf': [2, 3, 4],
              'max_features': [5, 3, 2],
            'n_estimators':[200,400,600]
}   

gbm = GradientBoostingClassifier()
gs_gbm = GridSearchCV(gbm, param_grids, n_jobs=-1, verbose=10).fit(X_train, y_train)
print(gs_gbm.best_params_)
```

    /anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.


    Fitting 3 folds for each of 540 candidates, totalling 1620 fits


    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:    3.6s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:   10.8s
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:   14.4s
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:   23.8s
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:   29.1s
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed:   35.7s
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed:   42.9s
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed:   49.3s
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed:   56.6s
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed:  1.7min
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed:  1.9min
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed:  2.2min
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed:  2.7min
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed:  3.1min
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed:  3.5min
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed:  4.0min
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed:  4.5min
    [Parallel(n_jobs=-1)]: Done 297 tasks      | elapsed:  4.9min
    [Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed:  5.3min
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed:  5.7min
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed:  6.0min
    [Parallel(n_jobs=-1)]: Done 405 tasks      | elapsed:  6.3min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed:  6.8min
    [Parallel(n_jobs=-1)]: Done 465 tasks      | elapsed:  7.2min
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed:  7.8min
    [Parallel(n_jobs=-1)]: Done 529 tasks      | elapsed:  8.6min
    [Parallel(n_jobs=-1)]: Done 562 tasks      | elapsed:  9.3min
    [Parallel(n_jobs=-1)]: Done 597 tasks      | elapsed: 10.5min
    [Parallel(n_jobs=-1)]: Done 632 tasks      | elapsed: 11.4min
    [Parallel(n_jobs=-1)]: Done 669 tasks      | elapsed: 12.2min
    [Parallel(n_jobs=-1)]: Done 706 tasks      | elapsed: 12.6min
    [Parallel(n_jobs=-1)]: Done 745 tasks      | elapsed: 13.3min
    [Parallel(n_jobs=-1)]: Done 784 tasks      | elapsed: 14.0min
    [Parallel(n_jobs=-1)]: Done 825 tasks      | elapsed: 14.7min
    [Parallel(n_jobs=-1)]: Done 866 tasks      | elapsed: 15.8min
    [Parallel(n_jobs=-1)]: Done 909 tasks      | elapsed: 17.3min
    [Parallel(n_jobs=-1)]: Done 952 tasks      | elapsed: 19.3min
    [Parallel(n_jobs=-1)]: Done 997 tasks      | elapsed: 20.3min
    [Parallel(n_jobs=-1)]: Done 1042 tasks      | elapsed: 20.8min
    [Parallel(n_jobs=-1)]: Done 1089 tasks      | elapsed: 21.4min
    [Parallel(n_jobs=-1)]: Done 1136 tasks      | elapsed: 22.1min
    [Parallel(n_jobs=-1)]: Done 1185 tasks      | elapsed: 23.3min
    [Parallel(n_jobs=-1)]: Done 1234 tasks      | elapsed: 25.0min
    [Parallel(n_jobs=-1)]: Done 1285 tasks      | elapsed: 26.8min
    [Parallel(n_jobs=-1)]: Done 1336 tasks      | elapsed: 27.6min
    [Parallel(n_jobs=-1)]: Done 1389 tasks      | elapsed: 28.3min
    [Parallel(n_jobs=-1)]: Done 1442 tasks      | elapsed: 29.0min
    [Parallel(n_jobs=-1)]: Done 1497 tasks      | elapsed: 30.4min
    [Parallel(n_jobs=-1)]: Done 1552 tasks      | elapsed: 31.9min
    [Parallel(n_jobs=-1)]: Done 1620 out of 1620 | elapsed: 34.5min finished


    {'learning_rate': 0.05, 'max_depth': 8, 'max_features': 5, 'min_samples_leaf': 3, 'n_estimators': 600}



```python
gs_gbm.score(X_test, y_test)
```




    0.5841750841750841



## XGBoost Classifier


```python
import xgboost as xgb
```


```python
data = xgb.DMatrix(X, target)
```


```python
param_grids = {'learning_rate': [0.1, 0.05, 0.2],
              'max_depth': [3, 4, 6],
              'min_samples_leaf': [2, 3, 4],
              'max_features': [5, 3, 2],
            'n_estimators':[200,400,600],
               'n_jobs':[-1,]
              }   

xgb_clf = xgb.XGBClassifier()
gs_xgb = GridSearchCV(xgb_clf, param_grids, n_jobs=-1, verbose=10).fit(X_train, y_train)
print(gs_xgb.best_params_)
```

    Fitting 3 folds for each of 243 candidates, totalling 729 fits


    /anaconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   2 tasks      | elapsed:   55.3s
    [Parallel(n_jobs=-1)]: Done   9 tasks      | elapsed:  2.9min
    [Parallel(n_jobs=-1)]: Done  16 tasks      | elapsed:  3.9min
    [Parallel(n_jobs=-1)]: Done  25 tasks      | elapsed:  6.8min
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  8.8min
    [Parallel(n_jobs=-1)]: Done  45 tasks      | elapsed: 11.6min
    [Parallel(n_jobs=-1)]: Done  56 tasks      | elapsed: 14.8min
    [Parallel(n_jobs=-1)]: Done  69 tasks      | elapsed: 17.8min
    [Parallel(n_jobs=-1)]: Done  82 tasks      | elapsed: 20.8min
    [Parallel(n_jobs=-1)]: Done  97 tasks      | elapsed: 25.8min
    [Parallel(n_jobs=-1)]: Done 112 tasks      | elapsed: 30.5min
    [Parallel(n_jobs=-1)]: Done 129 tasks      | elapsed: 36.2min
    [Parallel(n_jobs=-1)]: Done 146 tasks      | elapsed: 40.9min
    [Parallel(n_jobs=-1)]: Done 165 tasks      | elapsed: 47.0min
    [Parallel(n_jobs=-1)]: Done 184 tasks      | elapsed: 55.5min
    [Parallel(n_jobs=-1)]: Done 205 tasks      | elapsed: 64.5min
    [Parallel(n_jobs=-1)]: Done 226 tasks      | elapsed: 74.9min
    [Parallel(n_jobs=-1)]: Done 249 tasks      | elapsed: 83.4min
    [Parallel(n_jobs=-1)]: Done 272 tasks      | elapsed: 89.1min
    [Parallel(n_jobs=-1)]: Done 297 tasks      | elapsed: 95.1min
    [Parallel(n_jobs=-1)]: Done 322 tasks      | elapsed: 101.6min
    [Parallel(n_jobs=-1)]: Done 349 tasks      | elapsed: 109.9min
    [Parallel(n_jobs=-1)]: Done 376 tasks      | elapsed: 118.5min
    [Parallel(n_jobs=-1)]: Done 405 tasks      | elapsed: 127.8min
    [Parallel(n_jobs=-1)]: Done 434 tasks      | elapsed: 139.7min
    [Parallel(n_jobs=-1)]: Done 465 tasks      | elapsed: 152.4min
    [Parallel(n_jobs=-1)]: Done 496 tasks      | elapsed: 162.9min
    [Parallel(n_jobs=-1)]: Done 529 tasks      | elapsed: 171.4min
    [Parallel(n_jobs=-1)]: Done 562 tasks      | elapsed: 181.3min
    [Parallel(n_jobs=-1)]: Done 597 tasks      | elapsed: 192.2min
    [Parallel(n_jobs=-1)]: Done 632 tasks      | elapsed: 202.4min
    [Parallel(n_jobs=-1)]: Done 669 tasks      | elapsed: 216.1min
    [Parallel(n_jobs=-1)]: Done 706 tasks      | elapsed: 229.6min
    [Parallel(n_jobs=-1)]: Done 729 out of 729 | elapsed: 238.2min finished


    {'learning_rate': 0.2, 'max_depth': 6, 'max_features': 5, 'min_samples_leaf': 2, 'n_estimators': 600, 'n_jobs': -1}



```python
gs_xgb
```




    {'learning_rate': 0.2,
     'max_depth': 6,
     'max_features': 5,
     'min_samples_leaf': 2,
     'n_estimators': 600,
     'n_jobs': -1}




```python
gs_xgb.score(X_test, y_test)
```




    0.5942760942760943




```python
y_pred = gs_xgb.predict(X_test)
```


```python
confusion_matrix(y_test, y_pred)
```




    array([[352,  73, 170],
           [113, 223, 161],
           [120,  86, 484]])




```python
accuracy_score(y_test, y_pred)
```




    0.5942760942760943




```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               1       0.60      0.59      0.60       595
               2       0.58      0.45      0.51       497
               3       0.59      0.70      0.64       690
    
       micro avg       0.59      0.59      0.59      1782
       macro avg       0.59      0.58      0.58      1782
    weighted avg       0.59      0.59      0.59      1782
    


# Save Model


```python
import pickle
```


```python
# pickle.dump(gs_xgb, open('XGBoost_model.sav', 'wb'))

# # load the model from disk
# loaded_model = pickle.load(open('XGBoost_model.sav', 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)
```

    0.5942760942760943


# References

* <b>Peak Extraction Script.</b> Originally from https://github.com/libphy/which_animal, adapted to Python 3 and heavily tweaked for this study's purpose.
* <b>Leon Mak An Sheng, Mok Wei Xiong Edmund.</b> Deep Learning Approach to Accent Classification. Stanford CS229 Machine Learning Project.

