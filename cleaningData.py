import pandas as pd
import numpy as np
from scipy.stats import zscore


legitime_polluters=pd.read_csv('output_All_users.csv', sep=",", header=None)
legitime_polluters.columns=  ['UserID', 'tNumerOfFollowings', 'tNumberOfFollowers', 'tNumberOfTweets',
       'tLengthOfScreenName', 'LengthOfDescriptionInUserProfile',
       'RatioFollowingFollowers', 'DureVieCompte', 'RatioTweetDureeDeVie',
       'StanderedDiviation', 'NumberOfTweetsPerDay', 'MoyenUrlTweet',
       'RatioUrlTweet', 'Ratio@Tweet', 'TimeBetweenTweet',
       'MaxTempsEntreDeuxTweet', 'class']

### Data Cleaning
legitime_polluters=legitime_polluters[~legitime_polluters.isin([ np.inf, -np.inf]).any(1)] # supprimer les valeuxrs infinis
legitime_polluters= legitime_polluters.drop_duplicates() # remove duplicates
print(legitime_polluters.isnull().sum().sum() )# knowing how many missing values in the data
legitime_polluters.fillna(legitime_polluters.median(), inplace=True)  #remplace the nan values with the median

## normalisation with zscore
legitime_polluters=legitime_polluters.apply(lambda x: zscore(x) if x.name != 'UserID'and x.name != 'class' else x)

legitime_polluters.to_csv(r'output_All_users_clean.csv' ,header=None, index=None) # put the clean data data in a csv file
