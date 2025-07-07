import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless systems
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
import os
import shutil

# Ensure data/ directory exists
os.makedirs("data", exist_ok=True)

# Define dataset path
dataset_path = "data/compas-scores-two-years.csv"

# Download dataset if not present
if not os.path.exists(dataset_path):
    print(f"Downloading dataset to {dataset_path}")
    import urllib.request
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    try:
        urllib.request.urlretrieve(url, dataset_path)
        print(f"Dataset downloaded to {dataset_path}")
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        print("Please manually download from: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv")
        print(f"And place it in {dataset_path}")
        exit()

# Verify dataset exists
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found")
    exit()

# Copy dataset to aif360 directory (workaround for CompasDataset)
aif360_data_path = "compas_env/lib/python3.12/site-packages/aif360/data/raw/compas/compas-scores-two-years.csv"
os.makedirs(os.path.dirname(aif360_data_path), exist_ok=True)
try:
    shutil.copy(dataset_path, aif360_data_path)
    print(f"Copied dataset to {aif360_data_path}")
except Exception as e:
    print(f"Failed to copy dataset to {aif360_data_path}: {e}")
    print(f"Ensure {dataset_path} is readable and you have write permissions for {os.path.dirname(aif360_data_path)}")
    exit()

# Load dataset using default CompasDataset
try:
    compas_data = CompasDataset()
    print("Dataset loaded successfully from aif360 default path")
except Exception as e:
    print(f"Error loading CompasDataset: {e}")
    print(f"Ensure compas-scores-two-years.csv is in {os.path.dirname(aif360_data_path)}")
    exit()

# Rest of the code
privileged_groups = [{'race': 1}]  # Caucasian
unprivileged_groups = [{'race': 0}]  # African-American

metric = BinaryLabelDatasetMetric(compas_data, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
di = metric.disparate_impact()
spd = metric.statistical_parity_difference()
print(f"Disparate Impact Ratio: {di:.3f} (Ideal: 1.0)")
print(f"Statistical Parity Difference: {spd:.3f} (Ideal: 0.0)")

train, test = compas_data.split([0.7], shuffle=True)
pred_dataset = test.copy()
pred_dataset.labels = test.labels
class_metric = ClassificationMetric(test, pred_dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
fpr_diff = class_metric.false_positive_rate_difference()
eod = class_metric.equal_opportunity_difference()
print(f"False Positive Rate Difference: {fpr_diff:.3f} (Ideal: 0.0)")
print(f"Equal Opportunity Difference: {eod:.3f} (Ideal: 0.0)")

df, _ = compas_data.convert_to_dataframe()
plt.figure(figsize=(10, 6))
for race in df['race'].unique():
    subset = df[df['race'] == race]
    plt.hist(subset['score_text'], alpha=0.5, label=f'Race: {race}', bins=['Low', 'Medium', 'High'])
plt.title('Risk Score Distribution by Race')
plt.xlabel('Risk Score')
plt.ylabel('Count')
plt.legend()
plt.savefig('risk_score_distribution.png')
plt.close()

fpr_priv = class_metric.false_positive_rate(privileged=True)
fpr_unpriv = class_metric.false_positive_rate(privileged=False)
plt.figure(figsize=(8, 5))
plt.bar(['Caucasian', 'African-American'], [fpr_priv, fpr_unpriv], color=['blue', 'orange'])
plt.title('False Positive Rates by Race')
plt.ylabel('False Positive Rate')
plt.savefig('fpr_by_race.png')
plt.close()

reweigh = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
compas_reweighed = reweigh.fit_transform(compas_data)
metric_reweighed = BinaryLabelDatasetMetric(compas_reweighed, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
print(f"Disparate Impact Ratio (After Reweighing): {metric_reweighed.disparate_impact():.3f}")