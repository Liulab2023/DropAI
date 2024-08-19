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
