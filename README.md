# AI-driven High-throughput Droplet Screening of Cell-free Gene Expression

This project contains the Python code used in the paper "AI-driven high-throughput droplet screening of cell-free gene expression." The repository includes multiple sets of code, each serving a distinct purpose in the study: 
- Data pre-processing
- Neural Network Model
- XGBoost
- Transfer Learning Combined with XGBoost Code

# System Requirements and Dependencies

## Operating Systems
- Ubuntu 20.04 LTS
- Windows 10/11
- macOS Big Sur and later

## Software Dependencies
- Python: Version 3.8 or higher
- MATLAB: Version R2023a or higher



#  Project Structure

- `/Clustering Cleaning & TSNE/cleaning.mlx`:Matlab Live scripts are used for data preprocessing, including data clustering, cleaning, and T-SNE.
- `/XGBoost & Transfer Learning/main.py`: The main script that executes the core algorithm.
- `/XGBoost & Transfer Learning/code_demo`: Contains demo and related code required for the demo.
- `/Selection/method2_1.h5`: A pretrained model used to test demo, compiled with tensorflow keras.
- `/Selection/Stage1-selection.ipynb`: Jupyter notebook for model building and training.
  
#  Requirements

You can install all dependencies with the following command:

pip install -r requirements.txt


# Tested Versions

- Python: 3.8, 3.9
- TensorFlow: 2.5.0, 2.6.0
- scikit-learn: 0.24.2
- Matlab: R2023a


# Typical Installation Time

Estimated Time: The installation typically takes 5-10 minutes on a standard desktop computer with a reliable internet connection.


#  Usage

Place your dataset in the appropriate directory.

Run the Clustering Cleaning & TSNE/cleaning.mlx file to perform data preprocessing.

Run the Selection/Stage1-selection.ipynb file to build and train the model.

Run the XGBoost & Transfer Learning/main.py file to build and train the model.



#  Examples

You can run the following demo code:

- `/Clustering Cleaning & TSNE/cleaning.mlx`

- `/XGBoost & Transfer Learning/code_demo/demo_transfer_learning.ipynb`

- `/XGBoost & Transfer Learning/code_demo/demo_XGBoost_model.ipynb`



# Results and Analysis
After running the codes, you will obtain relevant results and evaluations of the model.


# License
This project is open-sourced under the MIT License. For more details, see the LICENSE file.
