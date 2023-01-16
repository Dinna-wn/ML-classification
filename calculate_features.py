import re
import numpy as np
import pandas as pd

################################################# first file ########################################################
# Le rapport following / followers
dff= pd.read_csv("content_polluters_followings.txt", sep="\t", header=None)
dff.columns=['UserID','SerieOfFollowings']

### L’écart type des IDs numériques uniques des following (the standard deviation of unique numerical IDs of following)
# 06 L’écart type des IDs numériques uniques des following
dff1 = dff['SerieOfFollowings'].str.split(',',expand=True).astype(float).std(axis= 1).reset_index(name='StanderedDiviation')
dff['StanderedDiviation']= dff1['StanderedDiviation']
dff= dff.drop(['SerieOfFollowings'], axis=1)


################################################# second file ########################################################
# Le rapport following / followers
df= pd.read_csv("content_polluters.txt", sep="\t", header=None)
df.columns=['UserID','tCreatedAt','tCollectedAt','tNumerOfFollowings', 'tNumberOfFollowers', 'tNumberOfTweets',
            'tLengthOfScreenName', 'LengthOfDescriptionInUserProfile']
df['tCreatedAt'] = pd.to_datetime(df['tCreatedAt']).dt.date #supprimer le temps et laisser la date
df['tCollectedAt'] = pd.to_datetime(df['tCollectedAt']).dt.date #supprimer le temps et laisser la date
df["RatioFollowingFollowers"] = df["tNumerOfFollowings"]/df["tNumberOfFollowers"]

# dure de vi de compte et Le rapport nombre de tweets par rapport à la durée de vie du compte
df["DureVieCompte"] = (df["tCollectedAt"] - df["tCreatedAt"])
df['RatioTweetDureeDeVie']=df["tNumberOfTweets"]/ (df["DureVieCompte"].apply(lambda x: (x.total_seconds()/(3600*24))))

df['DureVieCompte'] =df["DureVieCompte"].astype(str).str.replace("days","") #remove the days unit from the column DureVieCompte
df['DureVieCompte'] =df["DureVieCompte"].astype('int64')
data1= df.drop(['tCreatedAt', 'tCollectedAt'], axis=1)


################################################# third file ########################################################

df= pd.read_csv("content_polluters_tweets.txt", sep="\t", header=None)
df.columns=['UserID','tTweetID','tTweet','tCreatedAt']
df1=df.copy()


 # Le nombre de tweets envoyé par jour
df1['tCreatedAt'] = pd.to_datetime(df1['tCreatedAt']).dt.date   #supprimer le temps et laisser la date
df2 = df1.groupby(['UserID','tCreatedAt']).size().reset_index(name='NumberOfTweetsPerDay')
df3 = df2.groupby('UserID').mean()
df3= df3.iloc[:-1 , :]  # supprimer la derniere ligne

# Le nombre total des tweets envoyés
df['tCreatedAt'] = pd.to_datetime(df['tCreatedAt'])
data= df.sort_values(by= ['UserID','tCreatedAt'])
df4= data.groupby('UserID')['tCreatedAt'].size().reset_index(name='NombreTotaleTweet')
df4 = df4.iloc[:-1 , :]  # drop the last line
df4= df4.drop(['UserID'], axis=1) # drop the column UserID
df3['NombreTotaleTweet'] = df4['NombreTotaleTweet'].values  # Concatinate the df3 $ df4

### Le nombre moyen d’URL par tweet
def urls  ( line):
   url = re.findall('https?://(?:[-\www.]|(?:%[\da-fA-F]{2}))+', str(line))
   occ = len(url)
   return occ
df['TweetURL']= df['tTweet'].apply(urls) #number of urls in a tweet
df5=df.groupby('UserID')['TweetURL'].agg(['mean']).rename(columns={'mean':'MoyenUrlTweet'}) ###### le ratio |TweetURL| / |Tweets|
df5 = df5.iloc[:-1 , :]
df3['MoyenUrlTweet'] = df5['MoyenUrlTweet'].values

## Le rapport des adresses URL par rapport au nombre de tweets : |TweetURL| / |Tweets|
df2= df.groupby(by=['UserID'], dropna=False).sum().rename(columns={'TweetURL':'TotalTweetUrl'})
df2 = df2.iloc[:-1 , :]
df3['TotalTweetUrl'] = df2['TotalTweetUrl'].values
df6= df3['TotalTweetUrl'].div(df3['NombreTotaleTweet']).reset_index(name='RatioUrlTweet')
df3['RatioUrlTweet'] = df6['RatioUrlTweet'].values


####Le rapport des mentions @ par rapport au nombre de tweets :
df['Tweet@']= df['tTweet'].str.count('@') #number of @ in a tweet
df2= df.groupby(by=['UserID'], dropna=False).sum()  ##### tweet @ by user
df7=df.groupby('UserID')['Tweet@'].agg(['mean']).rename(columns={'mean':'Ratio@Tweet'})
df7 = df7.iloc[:-1 , :]
df3['Ratio@Tweet'] = df7['Ratio@Tweet'].values

##### Le temps moyen (en minutes) entre deux tweets consécutifs
df['tCreatedAt']= pd.to_datetime(df['tCreatedAt'])
data= df.sort_values(by= ['UserID','tCreatedAt'])
data['diff']= data.groupby('UserID')['tCreatedAt'].diff().apply(lambda x : x/ np.timedelta64(1, 'm')).astype('float') # .mean()
df8= data.groupby('UserID')['diff'].agg(['mean']).rename(columns={'mean':'TimeBetweenTweet'})
df8= df8.iloc[:-1 , :]
df3['TimeBetweenTweet'] = df8['TimeBetweenTweet'].values

##### Le max temps moyen (en minutes) entre deux tweets consécutifs
df9= data.groupby(['UserID'])['diff'].max().reset_index(name='MaxTempsEntreDeuxTweet')  #, sort=False) #
df9= df9.iloc[:-1 , :]
df3['MaxTempsEntreDeuxTweet'] = df9['MaxTempsEntreDeuxTweet'].values
print(df3.columns)
df3['class'] = 1
df3= df3.drop(['NombreTotaleTweet', 'TotalTweetUrl'], axis=1) # drop the unnecessary features

#merge the 3 data framed of the 3 files
data_merged= pd.merge(data1, dff, on='UserID', how='inner')
data2= pd.merge(data_merged, df3, on='UserID', how='inner') # data1 is the dataframe of the first file and df3 is the df from the second file
# put it into a csv file
data2.to_csv(r'D:\uqam\pycharmprojects\tweets\output_polluers.csv', header=None, index=None, sep=' ', mode='a')
