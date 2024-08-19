# DropAI

Code for "AI-driven high-throughput droplet screening of cell-free gene expression"<br>

Description of the files in this folder:<br>

task_1.csv      : Dataset file for combination optimization.<br>
	Col 0	: Average GFP value<br>
	Col 1	: Standard deviation<br>
	Col 2-13	: One-hot encoding<br>
	Col 14-	: Some data values from statistics<br>

method2_1.h5	: pretrained model, tensorflow 2.7-gpu<br>

Stage1-selection.ipynb: Jupyter notebook Python script using in this work<br>
	Note	: This script covers the training and validation of neural network models, as well as the use of models to predict scores.<br>
		  The script can be run under this relative path.<br>
		  This script contains sample data and the results of the last run.<br>
# AI-driven High-throughput Droplet Screening of Cell-free Gene Expression

This project contains the Python code used in the paper "AI-driven high-throughput droplet screening of cell-free gene expression." The code is designed to implement the construction of an XGBoost model. It includes data preprocessing and model construction, as well as the integration of feature-enhanced transfer learning techniques to combine transfer learning with the XGBoost algorithm.

# System Requirements
Software Dependencies
Operating Systems:
Ubuntu 20.04 LTS
Windows 10/11
macOS Big Sur and later

# Python
python:Version 3.8 or higher


#  Project Structure
- `/XGBoost & Transfer Learning/main.py`: The main script that executes the core algorithm.
- `/XGBoost & Transfer Learning/code_demo`: Contains demo and related code required for the demo.
- `method2_1.h5`: pretrained model, tensorflow 2.7-gpu
- `Stage1-selection.ipynb`: Jupyter notebook Python script using in this work
- `cleaning,mlx`: Matlab Live Scripts used to clustering, cleaning, and TSNE

#  Requirements
The following Python version and libraries are required to run the code:

numpy==1.23.1
pandas==1.4.4
tqdm==4.65.0
matplotlib==3.5.3
logomaker==0.8
seaborn==0.12.2
biotite==0.38.0
biopython==1.76
ipython==8.5.0
ipykernel==5.5.5

You can install all dependencies with the following command:

pip install -r requirements.txt

# Tested Versions
Python: 3.8, 3.9
TensorFlow: 2.5.0, 2.6.0
scikit-learn: 0.24.2

# Non-standard Hardware Requirements

# Typical Installation Time

Estimated Time: The installation typically takes 5-10 minutes on a standard desktop computer with a reliable internet connection.


#  Usage
Place your dataset in the appropriate directory.
Run the main.py file to build and train the  model.
python main.py


#  Examples
You can run the following demo code:
python code_main/demo.py

# Results and Analysis
After running the code, you will obtain relevant evaluations of the model's quality.


# License
This project is open-sourced under the MIT License. For more details, see the LICENSE file.
