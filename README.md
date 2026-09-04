\# Master's Thesis – Battery Remaining Useful Life



This repository contains the Python implementation developed as part of my Master's thesis on battery data analysis and RUL assessment.



The repository includes scripts for NMC battery data processing, feature generation and analysis, correlation analysis, dataset preparation, and deep-learning-based modelling using LSTM and Transformer architectures.



\## Repository Structure



\### `LSTM\_AllCon/` (AllCon -all Configurations)



Contains the scripts associated with the Long Short-Term Memory (LSTM) modelling workflow.



The modules in this directory cover dataset preparation, model training, and execution of the LSTM-based experiments.



\### `NMC\_Transformer/`



Contains the implementation of the Transformer-based modelling approach for NMC battery data.



This directory includes scripts for data preparation, feature handling, and training of the Transformer model.



\---



\## Main Scripts



\### `NMC\_Correlation\_SOH\_Features\_Matrix.py`



Performs correlation analysis between the extracted NMC battery features and battery State of Health (SOH).



The script is intended to support the identification and comparison of features that exhibit meaningful relationships with battery degradation and SOH.



\### `NMC\_dataset\_loader.py`



Provides the data-loading and preparation functionality used within the NMC battery analysis workflow.



It serves as an interface between the processed battery data and the subsequent machine-learning or deep-learning stages.



\### `NMC\_features\_plot.py`



Provides visualisation routines for analysing the extracted NMC battery features.



The generated plots support the examination of feature behaviour, trends, and their relationship with battery degradation.



\### `NMC\_gen\_features\_npz.py`



Generates and processes features from the NMC battery data and stores the resulting feature representation in NumPy NPZ format.



The generated feature files can subsequently be used for statistical analysis and model training without repeating the complete feature-generation process.



\### `NMC\_load\_npz\_feature\_file.py`



Loads previously generated NPZ feature files for inspection, verification, and further analysis.



This utility provides a convenient way to access the processed feature data generated during the feature-extraction stage.



\### `NMC\_Visualize\_TrainedModel.py`



Provides visualisation and evaluation functionality for trained machine-learning/deep-learning models.



The script is intended to support the interpretation and comparison of model predictions and corresponding evaluation results.



\---



\## Analysis Workflow



The overall workflow represented by the repository can be summarised as follows:



1\. Preparation and loading of NMC battery data.

2\. Generation of battery-health-related features.

3\. Storage of processed features in NPZ format.

4\. Visualisation and analysis of extracted features.

5\. Correlation analysis between the extracted features and battery SOH.

6\. Preparation of processed data for model training.

7\. Training and evaluation of LSTM and Transformer-based models.

8\. Visualisation and interpretation of model performance.



\---



\## Data and Model Outputs



Raw battery datasets, trained model checkpoints, and large intermediate result files are not included in this repository.



The repository is intended primarily to provide the source code required to reproduce the data-processing, feature-analysis, and modelling workflow.



\---



\## Software



The analysis is implemented in Python and uses scientific computing, data-analysis, visualisation, and deep-learning libraries.



Specific dependencies can be identified from the imports used within the individual scripts.



\---



\## Author



Hardik Riziya

