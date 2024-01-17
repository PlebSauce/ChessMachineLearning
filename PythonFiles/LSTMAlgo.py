import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn import linear_model, model_selection, metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

csv_file_path = "C:/Users/jorda/PycharmProjects/ChessMachineLearning/CSV/games.csv"
df = pd.read_csv(csv_file_path)
columns_used = ['winner', 'white_id', 'white_rating', 'black_id', 'black_rating', 'moves']
df_used = df[columns_used]

onehotencoder = OneHotEncoder(sparse_output=False)
df_encoded = pd.get_dummies(df_used, columns=['winner', 'white_id', 'black_id', 'moves'])

df_used.loc[:,['player']] = df_used['white_id'].fillna(df_used['black_id'])
df_used.drop(['white_id', 'black_id'], axis=1, inplace=True)

grouped_by_player = df_used.groupby('player')

snip_length = 10
df['Sequence'] = df.groupby('player')['moves'].transform(lambda x: x.tolist())
df['NextMove'] = df.groupby('player')['moves'].shift(-1)

df.dropna(subset=['NextMove'], inplace=True)

df['SnippedSequence'] = df['Sequence'].apply(lambda x: x[:snip_length])
mylog_model = linear_model.LogisticRegression()

models = {}
for player, data in grouped_by_player:
    X = data.drop(['player', 'NextMove'], axis=1)
    Y = data['NextMove']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2)

    player_model = linear_model.LogisticRegression()
    player_model.fit(X_train, Y_train)

    models[player] = player_model

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