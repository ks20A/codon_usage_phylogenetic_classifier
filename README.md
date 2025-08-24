Project aim: 
This project involves the development of a neural network model to predict the phylogenetic classification of organisms based on their codon usage frequencies. 
Leveraging a dataset from a 2023 Nature study by Hallee and Khomtchouk et al. (https://www.nature.com/articles/s41598-023-28965-7), the model was trained on data from 12,964 organisms across 11 distinct kingdoms, successfully classifying organisms with high accuracy.

Data source: 
  CSV dataset from Hallee and Khomtchouk et al., 2023 that was processed using Python: 'codon_usage.csv'
  Please note: In the CSV file provided by Hallee and Khomtchouk et al., 2023. There was 2 columns where the name was divided across 2 columns, which resulted in the data being a column short. When the data was loaded into pandas, the pandas dataframe contain an empty cell (NaN) so I found the location of NaN and then deleted the column manually. 

11 distinct kingdoms utilised in paper: Archaea, viral, plantae, phage, vertebrate, mammalia, bacteria, primate, plasmodia, Invertebrate and rodentia

In codon_ML_project.py
  Data was pre-processed and partitioned, with 80% of the dataset used for training the model and the remaining 20% reserved for testing the model accuracy. 
  The model was built using a sequential neural network with a dense layer containing 50 neurons, a final output layer of 11 neurons for classification and it was trained over 10 epochs. This model had only a 53% accuracy so it was further optimised. 