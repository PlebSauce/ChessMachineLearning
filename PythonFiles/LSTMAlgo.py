import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn import linear_model, model_selection, metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, LabelEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten

#csv_file_path = "C:/Users/jorda/PycharmProjects/ChessMachineLearning/CSV/games.csv"
csv_file_path = "C:/Users/jorda/Downloads/ChessMachineLearning/CSV/Test50sample.csv"
df = pd.read_csv(csv_file_path)
columns_used = ['winner', 'white_id', 'white_rating', 'black_id', 'black_rating', 'moves']
df_used = df[columns_used].copy().reset_index(drop=True)

df_used['player'] = df_used['white_id'].fillna(df_used['black_id'])
df_used['player'].fillna(df_used['white_id'], inplace=True)

df_used.drop(columns=['white_id', 'black_id'], axis=1, inplace=True)

df_used.groupby('player')

snip_length = 10
df_used['moves'] = df_used['moves'].apply(lambda x: x.split())

min_games = 3
grouped_by_player = df_used.groupby('player')
valid_players = [player for player, data in grouped_by_player if len(data) >= min_games]
newg = df_used[df_used['player'].isin(valid_players)]
newgrouped = newg.groupby('player')

mylog_model = linear_model.LogisticRegression()
le = LabelEncoder()
onehotencoder = OneHotEncoder(sparse_output=False)
models = {}
X_test_models = {}
Y_test_models = {}

for player, data in newgrouped:
    data.drop(['player'], axis=1, inplace=True)
    data['Sequence'] = data['moves'].apply(lambda x: ' '.join(x))
    data['SnippedSequence'] = data['Sequence'].apply(lambda x: ' '.join(x.split()[:snip_length]))
    data.drop(['Sequence', 'moves'], axis=1, inplace=True)

    data.loc[:, ['NextMove']] = ""
    count_of_rows = 0
    unique_moves = []
    for index, row in data.iterrows():
        arr = row['SnippedSequence']
        row['NextMove'] = arr.rpartition(' ')[-1]
        data.loc[index, 'NextMove'] = row['NextMove']
        if data.loc[index, 'NextMove'] not in unique_moves:
            unique_moves.append(data.loc[index, 'NextMove'])
            count_of_rows = count_of_rows + 1
    data.dropna(subset=['NextMove'], inplace=True)

    data['winner'] = le.fit_transform(data['winner'] )
    data['SnippedSequence'] = le.fit_transform(data['SnippedSequence'])
    data['NextMove'] = le.fit_transform(data['NextMove'])
    df_encoded = pd.get_dummies(data, columns=['winner', 'SnippedSequence', 'NextMove'])
    

    X = df_encoded.values[:, :(count_of_rows)]
    Y = df_encoded.values[:, :-count_of_rows]
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, train_size=0.8)

    onehotencoder.fit(Y_train.reshape(-1, 1))
    
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_train_reshaped = tf.constant(X_train_reshaped, dtype=tf.float32) 
    X_test_reshaped = tf.constant(X_test_reshaped, dtype=tf.float32)
    Y_train = tf.constant(Y_train, dtype=tf.float32)
    Y_test = tf.constant(Y_test, dtype=tf.float32)

    player_model = Sequential()
    player_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    player_model.add(Flatten())
    player_model.add(Dense(units=Y_train.shape[1], activation = 'softmax'))

    player_model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    prediction = player_model.predict(X_test_reshaped)

    player_model.fit(X_train_reshaped, Y_train, verbose=2, batch_size=1, epochs=50, validation_data=(X_test_reshaped, Y_test)) #may need validation_split=0.2

    X_test_models[player] = X_test_reshaped
    Y_test_models[player] = Y_test
    models[player] = player_model

player_ids_list = list(models.keys())
print("List of Player IDs:")
for index, player_ids in enumerate(player_ids_list, 1):
    print(f"{index}. {player_ids}")

selected_index = int(input("Enter the index ID of the player you would like to analyze:"))-1

if 0 <= selected_index < len(player_ids_list):
    selected_player = player_ids_list[selected_index]
    player_model = models[selected_player]
    player_x_test = X_test_models[selected_player]
    player_y_test = Y_test_models[selected_player]

    prediction = player_model.predict(player_x_test)
    column_names = onehotencoder.get_feature_names_out()

    relevant_columns = column_names[:prediction.shape[1]]
    prediction = prediction.reshape(-1, len(relevant_columns))
    predictions = np.argmax(prediction, axis=1)
    predictions = le.inverse_transform(predictions)

    decoded_predictions_index = np.argmax(prediction, axis=1)
    decoded_predictions_label = column_names[decoded_predictions_index]
    
    print(column_names)                                     
    #decoded_predictions = onehotencoder.inverse_transform(prediction)
    print("prediction:")
    print(prediction)
    print("decoded_predictions:")
    print(decoded_predictions_label)
    print("predictions:")
    print(predictions)
    accuracy = accuracy_score(np.argmax(player_y_test, axis=1), predictions)

    print(accuracy)
    #print(metrics.accuracy_score(player_y_test, prediction))
    #print(metrics.mean_squared_error(player_y_test, prediction))
else:
    print("Invalid ID")