import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from xgboost import XGBClassifier
import os

df = pd.read_csv("/content/TSHR PCR  results.csv")

columns_to_drop = ['Family H/o Infertility/Subfertility', 'Psychological/Occupational Stress',
                   'Parental consanguinity', 'Dental caries/Plaque', 'Sleep quantity']
df.drop(columns=columns_to_drop, inplace=True)

categorical_columns = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column].astype(str))

control_samples = df[df['Sl. No'].isin([1, 5])]
condition_samples = df[~df['Sl. No'].isin([1, 5])]
control_value = control_samples['2,-∆∆CT'].mean()

upper_threshold = 1.050011385
lower_threshold = 0.9424525244
condition_samples['Regulation'] = np.where(condition_samples['2,-∆∆CT'] > upper_threshold, 'Upregulated',
                                           np.where(condition_samples['2,-∆∆CT'] < lower_threshold, 'Downregulated', 'Neutral'))

correlation_matrix = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix', fontsize=18)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), bbox_inches='tight')  # Save the correlation matrix as an image with tight bounding box
plt.close() 

features = ['Age', 'Sex', 'Diabetes', 'Prediabetic', 'Hypertension', 'Dyslipidemia', 'CAD', 
            'CR attended', 'Sugar intake', 'Salt Intake', 'Type of Oil used', 'Frequency of fruit consumption', 
            'Water intake per day', 'Main source of drinking water', 'Frequency of dietary vegetable intake', 
            'frequency of leafy vegitables consumption', 'Regular food from home']

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

for feature in features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=feature, y='2,-∆∆CT', data=condition_samples, hue='Regulation', palette='coolwarm', split=True)
    plt.title(f'Violin Plot of 2,-∆∆CT Values by {feature}')
    plt.xlabel(feature)
    plt.ylabel('2,-∆∆CT')
    plt.legend(title='Regulation')
    plt.savefig(os.path.join(output_dir, f'{feature}_violin_plot.png'), bbox_inches='tight')
    plt.close()

X = df.drop(columns=['2,-∆∆CT'])
y = df['2,-∆∆CT']
num_bins = 3
bin_labels = range(num_bins)
y_binned = pd.cut(y, bins=num_bins, labels=bin_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred_binned = model.predict(X_test)

y_pred_continuous = y_pred_binned * (y.max() - y.min()) / (num_bins - 1) + y.min()
mse = mean_squared_error(y_test.astype(float), y_pred_continuous.astype(float))
print("Mean Squared Error:", mse)
print("Classification Report:")
print(classification_report(y_test, y_pred_binned))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_binned))
