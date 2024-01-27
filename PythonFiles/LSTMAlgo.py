import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
#df_used['player'] = df_used['player'].fillna(df_used['white_id'], inplace=True)

df_used.drop(columns=['white_id', 'black_id'], axis=1, inplace=True)

df_used.groupby('player')

snip_length = 10
df_used['moves'] = df_used['moves'].apply(lambda x: x.split())

min_games = 20

grouped_by_player = df_used.groupby('player')
valid_players = [player for player, data in grouped_by_player if len(data) >= min_games]
newg = df_used[df_used['player'].isin(valid_players)]
newgrouped = newg.groupby('player')

#mylog_model = linear_model.LogisticRegression()
le = LabelEncoder()
models = {}
X_train_models = {}
Y_train_models = {}
X_test_models = {}
Y_test_models = {}
predictions = {}
decoded_predictions = {}

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

    #data['winner'] = le.fit_transform(data['winner'] )
    #data['SnippedSequence'] = le.fit_transform(data['SnippedSequence'])
    #data['NextMove'] = le.fit_transform(data['NextMove'])
    data['winner'] = le.fit_transform(data['winner'])
    data['black_rating'] = le.fit_transform(data['black_rating'])
    data['white_rating'] = le.fit_transform(data['white_rating'])
    data['SnippedSequence'] = le.fit_transform(data['SnippedSequence'])
    #data['NextMove'] = le.fit_transform(data['NextMove'])
    
    X = data.iloc[:, :-1].values
    Y = le.fit_transform(data['NextMove']).reshape(-1, 1)
    #Y = data.iloc[:, -1].values.reshape(-1, 1)
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, train_size=0.8)
    #X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=1/len(X), random_state=42)
    
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
    

    player_model.fit(X_train_reshaped, Y_train, verbose=2, batch_size=1, epochs=50, validation_data=(X_test_reshaped, Y_test)) #may need validation_split=0.2

    X_train_models[player] = X_train_reshaped
    Y_train_models[player] = Y_train
    X_test_models[player] = X_test_reshaped
    Y_test_models[player] = Y_test
    models[player] = player_model

    prediction = player_model.predict(X_test_reshaped)

    print("decoded:")
    prediction = prediction.ravel()
    prediction_readable = le.inverse_transform(prediction.astype(int))
    print(prediction_readable)
    predictions[player] = prediction
    decoded_predictions[player] = prediction_readable

player_ids_list = list(models.keys())
print("List of Player IDs:")
for index, player_ids in enumerate(player_ids_list, 1):
    print(f"{index}. {player_ids}")

while (input("Would you like to select a player? (Y/N)")) != 'N':
    try:
        selected_index = int(input("Enter the index ID of the player you would like to analyze:"))-1
    except:
        ("Not an integer! Please try again.")
    if 0 <= selected_index < len(player_ids_list):
        selected_player = player_ids_list[selected_index]
        player_model = models[selected_player]
        player_x_train = X_train_models[selected_player]
        player_y_train = Y_train_models[selected_player]
        player_x_test = X_test_models[selected_player]
        player_y_test = Y_test_models[selected_player]
        player_prediction = predictions[selected_player]
        player_decoded_prediction = decoded_predictions[selected_player]

        print(player_y_test)
        print(player_prediction)
        print("predictions:")
        print(player_decoded_prediction)
        print("Accuracy:")
        print(metrics.accuracy_score(player_y_test, player_prediction))
    else:
        print("Invalid ID")
        #WARNING:tensorflow:6 out of the last 6 calls to 
        #<function Model.make_predict_function.<locals>.predict_function at 0x000002310FB73E20> 
        #triggered tf.function retracing. Tracing is expensive and the excessive number of 
        #tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing 
        #tensors with different shapes, (3) passing Python objects instead of tensors. 
        #For (1), please define your @tf.function outside of the loop. For (2), @tf.function has 
        #reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer 
        #to https://www.tensorflow.org/guide/function#controlling_retracing and 
        #https://www.tensorflow.org/api_docs/python/tf/function for  more details.