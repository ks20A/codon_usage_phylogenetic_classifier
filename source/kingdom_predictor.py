from keras.utils import to_categorical
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import keras_tuner as kt
from keras.callbacks import EarlyStopping

""""
Please Note: Had to remove 2 columns from the orginal data from the paper (Hallee and Khomtchouk et al., 2023). 
They both contained a name that was divided across 2 columns, which resulted in the data being a column short.
When the data was loaded into pandas, the pandas dataframe contain an empty cell (NaN) so found location of NaN and then deleted the column manually. 
No NaN values are present in codon_usage.csv file in this repo. 
"""

kingdom_dict = {
    'arc': 0, #Archaea
    'vrl': 1, #viral or virus
    'pln': 2, #Plant
    'phg': 3, #Phage or bacteriophage
    'vrt': 4, #vertebrate
    'mam': 5, #Mammal
    'bct': 6, #Bacteria
    'pri': 7, #Primate
    'plm': 8, #Plasmid
    'inv': 9, #Invertebrate
    'rod': 10 #Rodent
}

def get_pre_processed_data(csv_file_path):
    df = pd.read_csv(csv_file_path, sep=",")
    df = df.sample(frac=1) #to randomise data, 1 = all
  
    kingdom_list = df["Kingdom"].values.tolist()
    codon_df = df.iloc[:,5:]

    def kingdom_mapper(kingdom_text):
        global kingdom_dict
        return kingdom_dict[kingdom_text]

    kingdom_list_mapped = list(map(kingdom_mapper, kingdom_list))
    kingdoms_one_hot_labels = to_categorical(kingdom_list_mapped)
    
    return codon_df, kingdoms_one_hot_labels

def partition_data(codon_df, labels_1_hot):
    inputs = (codon_df * 1000).astype(float)
    n_training_samples = int(len(labels_1_hot) * 0.8) # Calculates the number of training samples by taking 80% of the total number of samples.

    training_inputs, training_labels = inputs[0:n_training_samples], labels_1_hot[0:n_training_samples] # Creates training data by taking the first 80% of the inputs and their corresponding labels.
    testing_inputs, testing_labels = inputs[n_training_samples:], labels_1_hot[n_training_samples:] # Creates testing data by taking the remaining 20% of the inputs and their corresponding labels.

    return training_inputs, training_labels, testing_inputs, testing_labels


def model_builder(hp):
    model = Sequential()
    model.add(Dense(50, input_dim=64, activation='relu'))
    hp_activation = hp.Choice('activations', values=["relu", "sigmoid", "tanh", "selu"])
    hp_layers = hp.Int('number_of_layers', min_value=1, max_value=20) #default of 1 for step
    for i in range(hp_layers): #adding certain number of hidden layers, pick between 1 and 20 and test between 1 and 100 nodes for each of 20 layers
        model.add(Dense(units=hp.Int(f"units_{i}", min_value=1, max_value=1000, step=100), activation=hp_activation))

    model.add(Dense(11, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    csv_file_path = "data/codon_usage.csv"
    # Data Preprocessing
    codon_df, labels_1_hot = get_pre_processed_data(csv_file_path)

    # Data Partitioning
    training_inputs, training_labels, testing_inputs, testing_labels = partition_data(codon_df, labels_1_hot)

    # Identify model hyperparameters 
    tuner = kt.Hyperband(model_builder, objective="val_accuracy", max_epochs=10, factor=3, directory="dir", project_name="simple_range_1_20_hidden_layers" ) #factor helps hyperparameter tuning
    stop_early = EarlyStopping(monitor="val_loss", patience=3) #patience is number of Epochs before determining if it improves #early stopper - reduce epochs if not improving
    tuner.search(training_inputs, training_labels, epochs=10, validation_data=(testing_inputs, testing_labels), callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters()[0]

    # Build model based on hyperparameters identified
    model = tuner.hypermodel.build(best_hps)
    print("Model Summary")
    print(model.summary())
    model.fit(training_inputs, training_labels, epochs=10, validation_data=(testing_inputs, testing_labels), callbacks=[stop_early])
    model.save("models/codon_range_1_20_hidden_layer2.keras")

if __name__ == "__main__":
    main()