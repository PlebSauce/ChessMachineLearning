import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn import linear_model, model_selection, metrics
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

csv_file_path = "C:/Users/jorda/PycharmProjects/ChessMachineLearning/CSV/games.csv"
df = pd.read_csv(csv_file_path)
columns_used = ['winner', 'white_id', 'white_rating', 'black_id', 'black_rating', 'moves']
df_used = df[columns_used].copy().reset_index(drop=True)

df_used['player'] = df_used['white_id'].fillna(df_used['black_id'])
df_used['player'].fillna(df_used['white_id'], inplace=True)

df_used.drop(columns=['white_id', 'black_id'], axis=1, inplace=True)

df_used.groupby('player')

snip_length = 10
df_used['moves'] = df_used['moves'].apply(lambda x: x.split())

min_games = 2
grouped_by_player = df_used.groupby('player')
valid_players = [player for player, data in grouped_by_player if len(data) >= min_games]
newg = df_used[df_used['player'].isin(valid_players)]
newgrouped = newg.groupby('player')

mylog_model = linear_model.LogisticRegression()
models = {}

for player, data in newgrouped:
    data.drop(['player'], axis=1, inplace=True)
    data['Sequence'] = data['moves'].apply(lambda x: ' '.join(x))
    data['SnippedSequence'] = data['Sequence'].apply(lambda x: ' '.join(x.split()[:snip_length]))
    data.drop(['Sequence', 'moves'], axis=1, inplace=True)

    arr = data['SnippedSequence'].values
    data.loc[:, ['NextMove']] = arr[0].rpartition(' ')[-1]
    data.dropna(subset=['NextMove'], inplace=True)

    onehotencoder = OneHotEncoder(sparse_output=False)
    df_encoded = pd.get_dummies(data, columns=['winner', 'SnippedSequence', 'NextMove'])
    #le = LabelEncoder()
    #df_encoded2 = le.fit_transform(df_used['NextMove'])
    X = df_encoded.values[:, [0, 1, 2, 4]]
    Y = df_encoded.values[:, 3]
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, train_size=0.8)

    #player_model = linear_model.LogisticRegression()
    #player_model.fit(X_train, Y_train)

    #models[player] = player_model

player_ids_list = list(models.keys())
print("List of Player IDs:")
for index, player_ids in enumerate(player_ids_list, 1):
    print(f"{index}. {player_ids}")

selected_index = int(input("Enter the index ID of the player you would like to analyze:"))-1

if 0 <= selected_index < len(player_ids_list):
    selected_player = player_ids_list[selected_index]
    player_model = models[selected_player]
    prediction = player_model.predict(X_test)
    print(metrics.accuracy_score(Y_test, prediction))
    print(metrics.mean_squared_error(Y_test, prediction))
else:
    print("Invalid ID")
