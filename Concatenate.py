import pandas as pd


# read the the data from the obtained files that we got from calculating the 15 features
data_legitime= pd.read_csv('output_legitime.csv', sep=" ", header=None)
data_polluters=pd.read_csv('output_polluers.csv', sep=" ", header=None)


concatenated = pd.concat([data_legitime, data_polluters], ignore_index=False) # concatinate the df legitime et df polluers
concatenated = concatenated.sample(frac=1, random_state=1).reset_index(drop=True) # shuffles rows
concatenated.to_csv(r'output_All_users.csv' ,header=None, index=None) # put the shuffled data in a csv file
