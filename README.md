Project Description
This project analyzes TSHR PCR results using machine learning techniques. The data preprocessing involves handling categorical variables, filtering control samples, and annotating condition samples based on threshold values. We visualize data correlations with a heatmap and explore features with violin plots. Finally, we build and evaluate an XGBoost classifier to predict the regulation of 2,-∆∆CT values, providing a classification report and confusion matrix for performance assessment.

Key Features
Data Preprocessing: Dropped irrelevant columns, encoded categorical variables, and calculated control sample values.
Condition Annotation: Classified samples as 'Upregulated', 'Downregulated', or 'Neutral' based on 2,-∆∆CT thresholds.
Correlation Analysis: Visualized feature correlations with a heatmap.
Feature Exploration: Generated violin plots to analyze 2,-∆∆CT distribution across various features.
Model Training: Used XGBoost to classify binned 2,-∆∆CT values and evaluated model performance.
For a detailed walkthrough, check out the Colab Notebook ------>>> https://shorturl.at/87q5g
