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


def main():
    csv_file_path = "data/codon_usage.csv"
    print(csv_file_path)
    codon_df, labels_1_hot = get_pre_processed_data(csv_file_path)
    training_inputs, training_labels, testing_inputs, testing_labels = partition_data(codon_df, labels_1_hot)


if __name__ == "__main__":
    main()