# DropAI

Code for "AI-driven high-throughput droplet screening of cell-free gene expression"

Description of the files in this folder:

task_1.csv		: Dataset file for combination optimization.
	Col 0	: Average GFP value
	Col 1	: Standard deviation
	Col 2-13	: One-hot encoding
	Col 14-	: Some data values from statistics

method2_1.h5	: pretrained model, tensorflow 2.7-gpu

Stage1-selection.ipynb: Jupyter notebook Python script using in this work
	Note	: This script covers the training and validation of neural network models, as well as the use of models to predict scores.
		  The script can be run under this relative path.
		  This script contains sample data and the results of the last run.
