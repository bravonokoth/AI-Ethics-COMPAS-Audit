# AI-Ethics-COMPAS-Audit
Bias audit for the COMPAS Recidivism Dataset using AI Fairness 360.

## Files
- `compas_audit.py`: Audit code and visualizations.
- `risk_score_distribution.png`: Risk score histogram.
- `fpr_by_race.png`: False positive rate bar plot.
- `compas_report.pdf`: 300-word report.
- `requirements.txt`: Dependencies.
- `.gitignore`: Excludes large files (e.g., `data/`, `compas_env/`).

## Setup
```bash
python3 -m venv compas_env
source compas_env/bin/activate
pip install -r requirements.txt

## Results
Disparate Impact Ratio: ~0.67 (bias detected).
False Positive Rate Difference: ~0.22 (higher for African-American defendants).
See compas_report.pdf for details.

##Dataset
The dataset (data/compas-scores-two-years.csv) is sourced from https://github.com/propublica/compas-analysis and excluded by .gitignore.

Create requirements.txt:
bash

pip freeze > requirements.txt
Verify contents:
bash

cat requirements.txt