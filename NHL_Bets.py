import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import openpyxl
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from dateutil.parser import parse

#path to data
hockey_games = 'C:/Users/Rob/Documents/GitHub/NHL_Bets/data/hockey_games.csv'
hockey_injuries = 'C:/Users/Rob/Documents/GitHub/NHL_Bets/data/hockey_injuries.csv'

# load the data.
df = pd.read_csv(hockey_games, skiprows=1, names=['date', 'visitor', 'visitor_goals', 'home', 'home_goals'])
df_home_inj = pd.read_csv(hockey_injuries)
df_visitor_inj = pd.read_csv(hockey_injuries)

# make the date column into a date format.
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df_home_inj['Date of Injury'] = df_home_inj['Date of Injury'].str[4:]
df_home_inj['Date of Injury'] = list(map(lambda x: datetime.strptime(x,'%b %d %Y').strftime('%Y-%m-%d'), df_home_inj['Date of Injury']))

df_visitor_inj['Date of Injury'] = df_visitor_inj['Date of Injury'].str[4:]
df_visitor_inj['Date of Injury'] = list(map(lambda x: datetime.strptime(x,'%b %d %Y').strftime('%Y-%m-%d'), df_visitor_inj['Date of Injury']))

df['goal_difference'] = df['home_goals'] - df['visitor_goals']

#new variables to show number of games in past 7 days for home/visitor -> NEED TO ADD LOGIC TO COUNT ONLY WITHIN DATE RANGE
df['7_days_prior'] = df['date'] - timedelta(days=7)
df['home_games_past_week'] = df.groupby('home')['home'].transform('count')
df['visitor_games_past_week'] = df.groupby('visitor')['visitor'].transform('count')

#new varaible to show number of injuries in previous 3 days -> NEED TO ADD LOGIC TO COUNT ONLY WITHIN DATE RANGE
df_home_inj['home_inj'] = df_home_inj.groupby('Team')['Team'].transform('count')
df_home_inj['home'] = df_home_inj['Team']
df_home_inj = df_home_inj.drop(columns=['Team','Date of Injury'])
df = pd.merge(df,df_home_inj.drop_duplicates(subset=['home']),how='left',on='home')

df_visitor_inj['visitor_inj'] = df_visitor_inj.groupby('Team')['Team'].transform('count')
df_visitor_inj['visitor'] = df_visitor_inj['Team']
df_visitor_inj = df_visitor_inj.drop(columns=['Team','Date of Injury'])
df = pd.merge(df,df_visitor_inj.drop_duplicates(subset=['visitor']),how='left',on='visitor')

df['home_inj'] = df['home_inj'].fillna(0)
df['visitor_inj'] = df['visitor_inj'].fillna(0)

df.to_excel('C:/Users/Rob/Documents/GitHub/NHL_Bets/data/df.xlsx', index=False)

# create new variables to show home team win or loss result
df['home_win'] = np.where(df['goal_difference'] > 0, 1, 0)
df['home_loss'] = np.where(df['goal_difference'] < 0, 1, 0)

df_visitor = pd.get_dummies(df['visitor'], dtype=np.int64) #need to add teams who have not played away or wait until later in season
df_home = pd.get_dummies(df['home'], dtype=np.int64) #need to add teams who have not played at home or wait until later in season

# subtract home from visitor
df_model = df_home.sub(df_visitor)

df_model['goal_difference'] = df['goal_difference']

df_train = df_model # not required but I like to rename my dataframe with the name train. ALSO NEED TO INCORPORATE NEW INJURY/FATIGUE DATA
df_train = df_train.fillna(0)

lr = Ridge(alpha=0.001) 
X = df_train.drop(['goal_difference'], axis=1)
y = df_train['goal_difference']

lr.fit(X, y)

df_ratings = pd.DataFrame(data={'team': X.columns, 'rating': lr.coef_})
print(df_ratings)