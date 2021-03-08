# Analysis of Local Field Potentials (LFPs) in Macaque Monkey's Primary Visual Cortex
## Project Summary
The aim of this project, which was completed as a capstone for Brown University's Master's Program in Data Science, is to better understand the dynamics of eye movements and fixations through analysis of neural data. In addition to granular data exploration, LSTM and GRU models were built to model LFP dynamics, decode types of eye fixations, and predict the occurrence of neural spikes. The full report can be found in **Capstone_Paper.pdf**.
## Data
The folder **data** contains the .csv files that were used to produce the results described in the report.
### Data Files
* **lfp.csv** contains LFP voltage data used to model LFP dynamics. This is referred to as **C1** in the full report.
* **lfp_mirror_saccade.csv** contains LFP voltage data used to decode types of eye fixations. This is referred to as **C2** in the full report.
* **spikes_lfp_full.csv** contains LFP voltage data used to predict the occurrence of neural spikes. This is referred to as **C3** in the full report.
## Models
### Saved Models
Trained Keras models are saved in the folders **lfp_models**, **mirror_saccade_models**, and **spike_models**.
### Training
**train_lfp.py**, **train_mirror_saccade.py**, and **train_spikes.py** can be used to reproduce the models above, as well as alter specifications and parameters. The models will be saved in the folders listed above.
## Error Analysis
Errors are stored in .npy files in the folders **lfp_errors** and **spike_errors**. Error analysis for the **mirror_saccade** models can be found in **mirror_decoding_errors_eda.ipynb**. 

To reproduce the error files above, run **lfp_error_analysis.py** and **spike_errors.py**.
## Plots and Tables
The plots and tables of raw data and model errors can be reproduced in the jupyter notebooks **lfp_prediction_errors_eda.ipynb**, **mirror_decoding_errors_eda.ipynb**, and **spike_prediction_errors_eda.ipynb**.

## Software Requirements
* TensorFlow
* Keras
* Pandas
* Matplotlib
* Python >= 3.6
* Seaborn
* Scikit-learn
* CUDA
* cuDNN

**Note:** any code that trains or directly uses any of the trained models must be run an a GPU, as Keras cudNNLSTM and cudNNGRU layers are not CPU-compatible.
