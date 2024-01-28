import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn import model_selection, metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.metrics import top_k_categorical_accuracy as keras_top_k

#csv_file_path = "C:/Users/jorda/PycharmProjects/ChessMachineLearning/CSV/games.csv"
csv_file_path = "C:/Users/jorda/Downloads/ChessMachineLearning/CSV/Test50sample.csv"
df = pd.read_csv(csv_file_path)
columns_used = ['winner', 'white_id', 'white_rating', 'black_id', 'black_rating', 'moves']
df_used = df[columns_used].copy().reset_index(drop=True)

le = LabelEncoder()

min_games = 10

df_used['player'] = df_used['white_id'].fillna(df_used['black_id'])

df_used.drop(columns=['white_id', 'black_id'], axis=1, inplace=True)

df_used.groupby('player')

snip_length = 10
df_used['moves'] = df_used['moves'].apply(lambda x: x.split())

df_used['Sequence'] = df_used['moves'].apply(lambda x: ' '.join(x))
df_used['SnippedSequence'] = df_used['Sequence'].apply(lambda x: ' '.join(x.split()[:snip_length]))
df_used.drop(['Sequence', 'moves'], axis=1, inplace=True)

df_used.loc[:, ['NextMove']] = ""
count_of_rows = 0
unique_moves = []
for index, row in df_used.iterrows():
    arr = row['SnippedSequence']
    row['NextMove'] = arr.rpartition(' ')[-1]
    df_used.loc[index, 'NextMove'] = row['NextMove']
    if df_used.loc[index, 'NextMove'] not in unique_moves:
        unique_moves.append(df_used.loc[index, 'NextMove'])
        count_of_rows = count_of_rows + 1
df_used.dropna(subset=['NextMove'], inplace=True)

unique_labels = df_used['NextMove'].unique()
le.fit(unique_labels)

grouped_by_player = df_used.groupby('player')
valid_players = [player for player, data in grouped_by_player if len(data) >= min_games]
newg = df_used[df_used['player'].isin(valid_players)]
newgrouped = newg.groupby('player')

models = {}
#accuracies = {}
k_accuracies = {}

k = int(input("Select the K value to be used for top K accuracy:"))

for player, data in newgrouped:
    data.drop(['player'], axis=1, inplace=True)
    data['winner'] = le.fit_transform(data['winner'])
    data['black_rating'] = le.fit_transform(data['black_rating'])
    data['white_rating'] = le.fit_transform(data['white_rating'])
    data['SnippedSequence'] = le.fit_transform(data['SnippedSequence'])

    X = data.iloc[:, :-1].values
    Y = le.fit_transform(data['NextMove']).reshape(-1, 1)
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, train_size=0.8)
    
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_train_reshaped = tf.constant(X_train_reshaped, dtype=tf.float32) 
    X_test_reshaped = tf.constant(X_test_reshaped, dtype=tf.float32)
    Y_train = tf.constant(Y_train, dtype=tf.float32)
    Y_test = tf.constant(Y_test, dtype=tf.float32)
    
    player_model = Sequential()
    player_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    player_model.add(Flatten())
    player_model.add(Dense(units=Y_train.shape[0] + 1, activation = 'softmax'))

    player_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy', keras_top_k])
    
    player_model.fit(X_train_reshaped, Y_train, verbose=2, batch_size=1, epochs=50, validation_data=(X_test_reshaped, Y_test)) #may need validation_split=0.2

    models[player] = player_model

    total_accuracy = 0 
    total_k_accuracy = 0
    for i in range(len(X_test_reshaped[0])):
    
        intest = X_test_reshaped[i:i+1]
        if intest.shape[0] == 0:
            print("No samples in the test set. Skipping prediction.")
            break
        prediction = player_model.predict(intest)
        #prediction_readable = le.inverse_transform([np.argmax(prediction)])
        #print("decoded:")
        #print(prediction_readable)

        true_label = Y_test[i:i+1]
        true_label_np = true_label.numpy().flatten().astype(int)
        true_label_readable = le.inverse_transform(true_label_np)
        print("True Value:")
        print(true_label_readable)
        #accuracy = metrics.accuracy_score(true_label_np, np.argmax(prediction, axis=1))
        #print(accuracy)

        #top_k_accuracy = keras_top_k(np.expand_dims(true_label_np, axis=0), np.expand_dims(prediction.astype(np.float32), axis=0), k=3)
        
        top_k_indices = np.argsort(prediction[0])[::-1][:k]
        top_k_predictions = le.inverse_transform(top_k_indices)
        print("Top {} Predictions:".format(k), top_k_predictions)        
        #print(f"Top-3 Accuracy: {top_k_accuracy.numpy()}")
        if true_label_readable[0] in top_k_predictions:
            top_k_accuracy = 1.0
        else:
            top_k_accuracy = 0.0
        print(top_k_accuracy)

        total_k_accuracy += top_k_accuracy
        #total_accuracy += accuracy
    #average_accuracy = total_accuracy / len(X_test_reshaped[0])
    average_k_accuracy = total_k_accuracy / len(X_test_reshaped[0])
    
    #accuracies[player] = average_accuracy
    k_accuracies[player] = average_k_accuracy

player_ids_list = list(models.keys())

while (input("Would you like to select a player? (Y/N)")) != 'N':
    try:
        print("List of Player IDs:")
        for index, player_ids in enumerate(player_ids_list, 1):
            print(f"{index}. {player_ids}")
        selected_index = int(input("Enter the index ID of the player you would like to analyze:"))-1
    except:
        ("Not an integer! Please try again.")
    if 0 <= selected_index < len(player_ids_list):
        selected_player = player_ids_list[selected_index]
        player_model = models[selected_player]
        #player_average_accuracy = accuracies[selected_player]
        player_average_k_accuracy = k_accuracies[selected_player]

        #print("Overall Accuracy:", player_average_accuracy)
        print("Overall top", k, "accuracy:", player_average_k_accuracy)
    else:
        print("Invalid ID")