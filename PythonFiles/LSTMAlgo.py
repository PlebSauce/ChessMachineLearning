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
from keras import callbacks
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten
from keras.metrics import top_k_categorical_accuracy as keras_top_k

tf.random.set_seed(65)

try:
    file_input = input("Enter file name in CSV file to use. No csv extension needed,"
                       " for example, for a file 'CSV/games.csv' type 'games':")
    csv_file_path = "CSV/" + file_input + ".csv"
except:
    print("Check your file pathing for errors")

#csv_file_path = "C:/Users/jorda/Downloads/ChessMachineLearning/CSV/Test50sample.csv"
#csv_file_path = "CSV/games.csv"
df = pd.read_csv(csv_file_path)
columns_used = ['winner', 'white_id', 'white_rating', 'black_id', 'black_rating', 'moves']
df_used = df[columns_used].copy().reset_index(drop=True)

le = LabelEncoder()
print("First you will select how many games a player needs to be considered in the dataset. "
      "Keep in mind higher values result in faster computing, but very few players to choose from "
      "and overfitting. Less players leads to slower computing times and less accurate data for players"
      " with low game counts.")
try:
    min_games = int(input("Select the minimum number of games a player needs (recommended 60):"))
    assert 1 <= min_games <= 200, "Out of bounds... Must be between 1-200 (inclusive)"
except ValueError as ve:
    print(f"Invalid input: {ve}")
except AssertionError as ae:
    print(f"Assertion error: {ae}")

dup_rows = df_used.copy()
df_used['player'] = df_used['white_id'].fillna(df_used['black_id'])

df_used.groupby('player')

snip_length = 11
df_used['moves'] = df_used['moves'].apply(lambda x: x.split())

df_used['Sequence'] = df_used['moves'].apply(lambda x: ' '.join(x))
df_used['SnippedSequence'] = df_used['Sequence'].apply(lambda x: ' '.join(x.split()[:snip_length]))

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
df_used['SnippedSequence'] = df_used['Sequence'].apply(lambda x: ' '.join(x.split()[:(snip_length-1)]))

dup_rows['winner'] = df_used['winner'].map({'white': 'black', 'black': 'white'})
dup_rows['white_id'], dup_rows['black_id'] = df_used['black_id'], df_used['white_id']
dup_rows['white_rating'], dup_rows['black_rating'] = df_used['black_rating'], df_used['white_rating']

dup_rows['player'] = dup_rows['white_id'].fillna(dup_rows['black_id'])
dup_rows.groupby('player')

snip_length = 10
dup_rows['moves'] = dup_rows['moves'].apply(lambda x: x.split())

dup_rows['Sequence'] = dup_rows['moves'].apply(lambda x: ' '.join(x))
dup_rows['SnippedSequence'] = dup_rows['Sequence'].apply(lambda x: ' '.join(x.split()[:snip_length]))

dup_rows.loc[:, ['NextMove']] = ""
count_of_rows = 0
unique_moves = []
for index, row in dup_rows.iterrows():
    arr = row['SnippedSequence']
    row['NextMove'] = arr.rpartition(' ')[-1]
    dup_rows.loc[index, 'NextMove'] = row['NextMove']
    if dup_rows.loc[index, 'NextMove'] not in unique_moves:
        unique_moves.append(dup_rows.loc[index, 'NextMove'])
        count_of_rows = count_of_rows + 1
dup_rows.dropna(subset=['NextMove'], inplace=True)
dup_rows['SnippedSequence'] = dup_rows['Sequence'].apply(lambda x: ' '.join(x.split()[:(snip_length-1)]))

df_final = pd.concat([df_used, dup_rows], ignore_index=True)

df_final.drop(['Sequence', 'moves'], axis=1, inplace=True)
df_final.drop(columns=['white_id', 'black_id'], axis=1, inplace=True)

unique_labels = df_final['NextMove'].unique()
le.fit(unique_labels)

grouped_by_player = df_final.groupby('player')
valid_players = [player for player, data in grouped_by_player if len(data) >= min_games]
newg = df_final[df_final['player'].isin(valid_players)]
newgrouped = newg.groupby('player')

models = {}
k_accuracies = {}
total = 0.0

try:
    k = int(input("Select the K value (recommended 3) to be used for top K accuracy between 1-10 (inclusive):"))
    assert 1 <= k <= 10, "Must be between 1-10 (inclusive)"
except ValueError as ve:
    print(f"Invalid input: {ve}")
except AssertionError as ae:
    print(f"Assertion error: {ae}")

for player, data in newgrouped:
    data.drop(['player'], axis=1, inplace=True)
    X_non_encoded = data.iloc[:, :-1].values
    Y_non_encoded = data['NextMove'].values

    X_train_non_encoded, X_test_non_encoded, Y_train_non_encoded, Y_test_non_encoded = model_selection.train_test_split(
        X_non_encoded, Y_non_encoded, test_size=0.2, train_size=0.8, random_state=65
    )
    data['winner'] = le.fit_transform(data['winner'])
    data['black_rating'] = le.fit_transform(data['black_rating'])
    data['white_rating'] = le.fit_transform(data['white_rating'])
    data['SnippedSequence'] = le.fit_transform(data['SnippedSequence'])

    X = data.iloc[:, :-1].values
    Y = le.fit_transform(data['NextMove']).reshape(-1, 1)
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, train_size=0.8, random_state=65)
    
    X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_train_reshaped = tf.constant(X_train_reshaped, dtype=tf.float32) 
    X_test_reshaped = tf.constant(X_test_reshaped, dtype=tf.float32)
    Y_train = tf.constant(Y_train, dtype=tf.float32)
    Y_test = tf.constant(Y_test, dtype=tf.float32)

    largest_index = np.max(Y)
    
    player_model = Sequential()
    player_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    player_model.add(Flatten())
    player_model.add(Dense(units=(largest_index + 1), activation = 'softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=65)))

    player_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy', keras_top_k])
    
    player_model.fit(X_train_reshaped, Y_train, verbose=2, batch_size=1, epochs=50, validation_data=(X_test_reshaped, Y_test))

    models[player] = player_model

    total_accuracy = 0 
    total_k_accuracy = 0
    print("For {}'s games".format(player))
    for i in range(len(X_test)):
        intest = X_test_reshaped[i:i+1]
        if intest.shape[0] == 0:
            print("No samples in the test set. Skipping prediction.")
            break
        prediction = player_model.predict(intest)

        intest_non_encoded = X_test_non_encoded[i:i+1]
        print("For move set")
        print(intest_non_encoded[:,3])

        true_label = Y_test[i:i+1]
        true_label_np = true_label.numpy().flatten().astype(int)
        true_label_readable = le.inverse_transform(true_label_np)
        print("Actual next move:")
        print(true_label_readable)

        top_k_indices = np.argsort(prediction[0])[::-1][:k]
        top_k_predictions = le.inverse_transform(top_k_indices)
        print("Top {} Predictions:".format(k), top_k_predictions)
        print("Accuracy: ")        
        if true_label_readable[0] in top_k_predictions:
            top_k_accuracy = 1.0
        else:
            top_k_accuracy = 0.0
        print(top_k_accuracy)

        total_k_accuracy += top_k_accuracy
    average_k_accuracy = total_k_accuracy / len(X_test)
    
    print("Average accuracy: ")
    print(average_k_accuracy)
    total += average_k_accuracy
    k_accuracies[player] = average_k_accuracy

player_ids_list = list(models.keys())
total /= len(player_ids_list)
print("Overall average", k, "accuracy:", total)

while True: 
    continue_input = (input("Would you like to select a player? (Y/N)"))
    if continue_input.lower() in ['yes', 'y']:   
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
            player_average_k_accuracy = k_accuracies[selected_player]

            print("Player top", k, "accuracy:", player_average_k_accuracy)
        else:
            print("Invalid ID")
    if continue_input.lower() in ["no", 'n']:
        print("Exiting...")
        break
    if continue_input.lower() not in ["yes", "no", 'y', 'n']:
        print("Enter Y or N")