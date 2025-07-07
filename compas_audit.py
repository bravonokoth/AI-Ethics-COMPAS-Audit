# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Load COMPAS dataset
try:
    compas_data = CompasDataset()
except Exception as e:
    print(f"Error loading CompasDataset: {e}")
    print("Ensure aif360 is installed and dataset is accessible.")
    exit()

privileged_groups = [{'race': 1}]  # Caucasian
unprivileged_groups = [{'race': 0}]  # African-American

# Compute fairness metrics
metric = BinaryLabelDatasetMetric(compas_data, 
                                 unprivileged_groups=unprivileged_groups, 
                                 privileged_groups=privileged_groups)
di = metric.disparate_impact()
spd = metric.statistical_parity_difference()
print(f"Disparate Impact Ratio: {di:.3f} (Ideal: 1.0)")
print(f"Statistical Parity Difference: {spd:.3f} (Ideal: 0.0)")

# Simulate predictions
train, test = compas_data.split([0.7], shuffle=True)
pred_dataset = test.copy()
pred_dataset.labels = test.labels  # Simulated predictions
class_metric = ClassificationMetric(test, pred_dataset, 
                                   unprivileged_groups=unprivileged_groups, 
                                   privileged_groups=privileged_groups)
fpr_diff = class_metric.false_positive_rate_difference()
eod = class_metric.equal_opportunity_difference()
print(f"False Positive Rate Difference: {fpr_diff:.3f} (Ideal: 0.0)")
print(f"Equal Opportunity Difference: {eod:.3f} (Ideal: 0.0)")

# Visualizations
df, _ = compas_data.convert_to_dataframe()

# Plot 1: Risk score distribution
plt.figure(figsize=(10, 6))
for race in df['race'].unique():
    subset = df[df['race'] == race]
    plt.hist(subset['score_text'], alpha=0.5, label=f'Race: {race}', bins=['Low', 'Medium', 'High'])
plt.title('Risk Score Distribution by Race')
plt.xlabel('Risk Score')
plt.ylabel('Count')
plt.legend()
plt.savefig('risk_score_distribution.png')
plt.show()

# Plot 2: False Positive Rates
fpr_priv = class_metric.false_positive_rate(privileged=True)
fpr_unpriv = class_metric.false_positive_rate(privileged=False)
plt.figure(figsize=(8, 5))
plt.bar(['Caucasian', 'African-American'], [fpr_priv, fpr_unpriv], color=['blue', 'orange'])
plt.title('False Positive Rates by Race')
plt.ylabel('False Positive Rate')
plt.savefig('fpr_by_race.png')
plt.show()

# Apply reweighing
reweigh = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
compas_reweighed = reweigh.fit_transform(compas_data)
metric_reweighed = BinaryLabelDatasetMetric(compas_reweighed, 
                                           unprivileged_groups=unprivileged_groups, 
                                           privileged_groups=privileged_groups)
print(f"Disparate Impact Ratio (After Reweighing): {metric_reweighed.disparate_impact():.3f}")