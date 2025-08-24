Project title: Codon Usage Phylogenetic Classifier

Introduction:
  This project involves the development of a neural network model to predict the phylogenetic classification of organisms based on their codon usage frequencies. 
  Leveraging a dataset from a 2023 Nature study by Hallee and Khomtchouk et al. (https://www.nature.com/articles/s41598-023-28965-7), the model was trained on data from 12,964 organisms across 11 distinct kingdoms, successfully classifying organisms with high accuracy. The goal is to demonstrate that codon usage contains enough information to accurately predict an organism's broad taxonomic classification.

Dataset
  The analysis leverages a dataset of 12,964 organisms, which includes their names, kingdom classifications and the frequencies of all 64 codons. 
  CSV dataset from Hallee and Khomtchouk et al., 2023 that was processed using Python and called 'codon_usage.csv'
  Please note: In the CSV file provided by Hallee and Khomtchouk et al., 2023. There was 2 columns where the name was divided across 2 columns, which resulted in the data being a column short. When the data was loaded into pandas, the pandas dataframe contain an empty cell (NaN) so I found the location of NaN and then deleted the column manually. 

This data spans 11 major kingdoms that were utilised in the paper: Archaea, viral, plantae, phage, vertebrate, mammalia, bacteria, primate, plasmodia, invertebrate and rodentia

Methodology
  Data was pre-processed and partitioned and the model was built in kingdom_predictor.py

  Data Preprocessing: 
  The raw CSV dataset was cleaned and processed to handle inconsistencies. The categorical kingdom labels were mapped to numerical values and then one-hot encoded for model training. The codon frequency data was normalised and converted to a float type.

  Data Partitioning: 
  The dataset was split into training and testing sets, with 80% of the data used for training and 20% used for validation, to ensure the model's performance could be accurately evaluated on unseen data.

  Hyperparameter Tuning and Evaluation: 
  A hyperparameter-tuned neural network was built and trained. The model's architecture was optimised to find the best configuration by exploring a range of options including:
    The number of hidden layers, from 1 to 20.
    The number of neurons in each layer, ranging from 1 to 1000.
    The activation function for the hidden layers, including relu, sigmoid, tanh, and selu.
  
  The best performing model was selected after a search, trained on 10 epochs and evaluated using accuracy and loss metrics on a dedicated testing dataset.
  Accuracy achieved: ~93%

The model was then saved in the models folder. 


How to Run the Project:
  Clone the repository: git clone https://github.com/ks20A/codon_usage_phylogenetic_classifier.git
  Create a virtual environment: python3 -m venv venv
  Activate the virtual environment: source venv/bin/activate 
  Install dependencies: pip3 install -r requirements.txt
  Run the script: python3 source/kingdom_predictor.py
