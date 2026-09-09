# Medical Insurance Analytics Dashboard

Lab-4 — Applied Statistical Modeling & Interactive Web Dashboard
Course: Statistical Modeling with Python (M.Sc. Data Science, Sem 1)

An interactive Streamlit dashboard covering exploratory analysis, hypothesis
testing, OLS regression, and residual diagnostics on the **Medical Insurance
Costs** dataset — the same analysis performed in the accompanying Jupyter
notebook, now delivered as a live web app.

## How to run

### Option A — Local IDE / VS Code (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### Option B — Google Colab (fallback)

```python
!pip install streamlit
```
```python
%%writefile app.py
# paste the contents of app.py here
```
```python
!streamlit run app.py & npx localtunnel --port 8501
```

## Data

Place `insurance.csv` inside a `data/` folder next to `app.py`
(`data/insurance.csv`). If no local copy is found, the app automatically
tries to download a public mirror of the dataset; if that also fails (e.g.
no internet access), it will prompt you to upload the CSV manually.

**Dataset summary** — 1,338 records, 7 columns:

| Column   | Type        | Description                              |
|----------|-------------|-------------------------------------------|
| age      | numeric     | Age of primary beneficiary                |
| sex      | categorical | male / female                             |
| bmi      | numeric     | Body mass index                           |
| children | numeric     | Number of dependents covered              |
| smoker   | categorical | yes / no                                  |
| region   | categorical | northeast / northwest / southeast / southwest |
| charges  | numeric     | Individual medical costs billed (target)  |

## Dashboard structure

- **Tab 1 — Data Exploration:** sidebar filters (age, BMI, children, sex,
  smoker, region), reactive Plotly histograms/scatter plots with trendlines,
  a correlation heatmap, and live summary statistics for the filtered slice.
- **Tab 2 — Hypothesis Testing Lab:** pick any categorical factor and
  numerical metric; the app runs Shapiro-Wilk and Levene's tests, then
  automatically selects a two-sample t-test, Mann-Whitney U test, or
  one-way ANOVA and reports the Reject/Fail-to-Reject decision at α = 0.05.
  A separate Chi-square section tests independence between any two
  categorical variables.
- **Tab 3 — Live Prediction & Diagnostics:** enter a hypothetical
  individual's profile to get a real-time OLS charge prediction with a 95%
  confidence interval (for the mean) and a 95% prediction interval (for a
  new individual), plus residuals-vs-fitted and Q-Q plots, a Jarque-Bera
  normality test, and VIF multicollinearity diagnostics.

## Synthesis of statistical findings (from the notebook)

- **Smokers vs. non-smokers:** Charges are non-normal in both groups
  (Shapiro-Wilk p < 0.05), so a Mann-Whitney U test was used. The result
  **rejects H₀** — smokers incur significantly higher medical charges than
  non-smokers.
- **Charges across regions:** A one-way ANOVA across the four regions
  **fails to reject H₀** — mean charges do not differ significantly by
  region.
- **Smoking status vs. region (Chi-square):** No statistically significant
  association was found between smoking status and region.
- **OLS regression** (`charges ~ age + bmi + children + sex + smoker +
  region`): `smoker_yes`, `age`, and `bmi` are highly significant positive
  predictors of charges; `sex` and `region` are largely non-significant.
  The model explains a substantial share of variance in charges (see the
  in-app R² / adjusted R²).
- **Diagnostics:** VIF values for all predictors are low (< 2), indicating
  no meaningful multicollinearity. The Jarque-Bera test indicates the
  residuals deviate from normality — expected given the right-skewed
  nature of medical charges — which is a limitation to note when using the
  model's prediction intervals.

## Files

```
.
├── app.py              # Streamlit dashboard (3 tabs)
├── requirements.txt    # Python dependencies
├── data/
│   └── insurance.csv   # dataset (add your own copy here)
└── README.md
```

## Optional: Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and select the repo + `app.py` as the entry point.
3. Add the resulting live URL to this README's header once deployed.
