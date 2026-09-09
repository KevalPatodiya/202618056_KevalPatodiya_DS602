"""
Lab-4: Applied Statistical Modeling & Interactive Web Dashboard
Dataset : Medical Insurance Costs (insurance.csv)
Author  : 202618056

Run with:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, f_oneway, chi2_contingency, jarque_bera

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Medical Insurance Analytics Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ALPHA = 0.05

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data(show_spinner=True)
def load_data():
    """Load the insurance dataset from a local file, falling back to a
    public mirror of the dataset if no local copy is found."""
    candidate_paths = [
        "data/insurance.csv",
        "insurance.csv",
        os.path.join(os.path.dirname(__file__), "data", "insurance.csv"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return pd.read_csv(path), f"local file ({path})"

    fallback_url = (
        "https://raw.githubusercontent.com/stedy/"
        "Machine-Learning-with-R-datasets/master/insurance.csv"
    )
    try:
        df = pd.read_csv(fallback_url)
        return df, "downloaded from public mirror"
    except Exception:
        return None, None


df, source = load_data()

if df is None:
    st.warning(
        "Couldn't find `data/insurance.csv` locally and couldn't reach the "
        "internet to download it. Please upload the dataset to continue."
    )
    uploaded = st.file_uploader("Upload insurance.csv", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        source = "uploaded by user"
    else:
        st.stop()

df = df.drop_duplicates().reset_index(drop=True)

numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()


# ============================================================
# MODEL FITTING (cached, mirrors the notebook's Part 4)
# ============================================================
@st.cache_resource(show_spinner=True)
def fit_ols_model(data: pd.DataFrame):
    df_model = pd.get_dummies(
        data, columns=["sex", "smoker", "region"], drop_first=True
    )
    # ensure boolean dummies behave as 0/1 ints for statsmodels
    dummy_cols = [c for c in df_model.columns if df_model[c].dtype == bool]
    df_model[dummy_cols] = df_model[dummy_cols].astype(int)

    y = df_model["charges"]
    X = df_model.drop(columns=["charges"])
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    return model, X, y, df_model.columns.tolist()


ols_model, X_design, y_target, model_columns = fit_ols_model(df)
feature_columns = [c for c in X_design.columns if c != "const"]


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("💊 Insurance Dashboard")
st.sidebar.caption(f"Data source: {source} • {df.shape[0]} rows")

st.sidebar.markdown("### Filters (Data Exploration tab)")

age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

bmi_min, bmi_max = float(df["bmi"].min()), float(df["bmi"].max())
bmi_range = st.sidebar.slider(
    "BMI range", round(bmi_min, 1), round(bmi_max, 1), (round(bmi_min, 1), round(bmi_max, 1))
)

sex_filter = st.sidebar.multiselect(
    "Sex", options=sorted(df["sex"].unique()), default=sorted(df["sex"].unique())
)
smoker_filter = st.sidebar.multiselect(
    "Smoker", options=sorted(df["smoker"].unique()), default=sorted(df["smoker"].unique())
)
region_filter = st.sidebar.multiselect(
    "Region", options=sorted(df["region"].unique()), default=sorted(df["region"].unique())
)
children_range = st.sidebar.slider(
    "Number of children",
    int(df["children"].min()),
    int(df["children"].max()),
    (int(df["children"].min()), int(df["children"].max())),
)

filtered_df = df[
    (df["age"].between(*age_range))
    & (df["bmi"].between(*bmi_range))
    & (df["sex"].isin(sex_filter))
    & (df["smoker"].isin(smoker_filter))
    & (df["region"].isin(region_filter))
    & (df["children"].between(*children_range))
]

st.title("Applied Statistical Modeling & Interactive Web Dashboard")
st.caption("Dataset: Medical Insurance Costs — EDA · Hypothesis Testing · OLS Regression & Diagnostics")

tab1, tab2, tab3 = st.tabs(
    ["📊 Data Exploration", "🧪 Hypothesis Testing Lab", "🔮 Live Prediction & Diagnostics"]
)

# ============================================================
# TAB 1 — DATA EXPLORATION
# ============================================================
with tab1:
    st.subheader("Filtered Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows selected", f"{len(filtered_df):,}")
    c2.metric("Avg. charges", f"${filtered_df['charges'].mean():,.0f}" if len(filtered_df) else "—")
    c3.metric("Avg. BMI", f"{filtered_df['bmi'].mean():.1f}" if len(filtered_df) else "—")
    c4.metric(
        "Smoker %",
        f"{(filtered_df['smoker'].eq('yes').mean() * 100):.1f}%" if len(filtered_df) else "—",
    )

    if filtered_df.empty:
        st.info("No rows match the current filters. Adjust the sidebar filters.")
    else:
        with st.expander("Summary statistics", expanded=True):
            st.dataframe(filtered_df.describe().T.round(2), use_container_width=True)

        with st.expander("Raw filtered data"):
            st.dataframe(filtered_df, use_container_width=True)

        st.markdown("### Distributions")
        dist_col = st.selectbox("Choose a numerical feature", numerical_columns, key="dist_col")
        fig_hist = px.histogram(
            filtered_df, x=dist_col, color="smoker", marginal="box", nbins=30,
            title=f"Distribution of {dist_col}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### Bivariate Relationships")
        col_a, col_b = st.columns(2)
        with col_a:
            x_axis = st.selectbox("X-axis", numerical_columns, index=0, key="x_axis")
        with col_b:
            y_axis = st.selectbox("Y-axis", numerical_columns, index=len(numerical_columns) - 1, key="y_axis")
        color_by = st.selectbox("Color by", categorical_columns, key="color_by")

        color_map = {"male": "#1f77b4", "female": "#e84f4f"}  # blue / pink

        fig_scatter = px.scatter(
            filtered_df, x=x_axis, y=y_axis, color=color_by, trendline="ols",
            color_discrete_map=color_map if color_by == "sex" else None,
            title=f"{x_axis} vs {y_axis} (colored by {color_by})",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Correlation Matrix")
        corr = filtered_df[numerical_columns].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlation Matrix (numerical features)",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("### Categorical Breakdown")
        cat_col = st.selectbox("Choose a categorical feature", categorical_columns, key="cat_col")
        fig_bar = px.histogram(filtered_df, x=cat_col, color=cat_col, title=f"Counts of {cat_col}")
        st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# TAB 2 — HYPOTHESIS TESTING LAB
# ============================================================
with tab2:
    st.subheader("Two-Group / Multi-Group Comparison")
    st.write(
        "Select a categorical factor and a numerical metric. The app automatically "
        "checks normality (Shapiro-Wilk) and equal variance (Levene), then runs the "
        "appropriate test: a two-sample t-test or Mann-Whitney U test for 2 groups, "
        "or a one-way ANOVA for 3+ groups."
    )

    colA, colB = st.columns(2)
    with colA:
        factor_col = st.selectbox("Categorical factor", categorical_columns, key="factor_col")
    with colB:
        metric_col = st.selectbox("Numerical metric", numerical_columns, index=len(numerical_columns) - 1, key="metric_col")

    groups = {level: df.loc[df[factor_col] == level, metric_col].dropna() for level in df[factor_col].unique()}
    group_names = list(groups.keys())

    st.markdown(f"**Groups found in `{factor_col}`:** {', '.join(group_names)}")

    normality_rows = []
    for name, values in groups.items():
        stat, p = shapiro(values) if len(values) >= 3 else (np.nan, np.nan)
        normality_rows.append({"Group": name, "n": len(values), "Shapiro W": stat, "p-value": p,
                                "Normal?": "Yes" if p > ALPHA else "No"})
    st.markdown("**Shapiro-Wilk normality check**")
    st.dataframe(pd.DataFrame(normality_rows).round(4), use_container_width=True)

    if len(group_names) == 2:
        g1, g2 = groups[group_names[0]], groups[group_names[1]]
        lev_stat, lev_p = levene(g1, g2)
        st.markdown(
            f"**Levene's test for equal variance:** statistic = {lev_stat:.4f}, "
            f"p-value = {lev_p:.4g} → variances are "
            f"{'equal' if lev_p > ALPHA else 'not equal'}."
        )

        both_normal = all(row["p-value"] > ALPHA for row in normality_rows)
        if both_normal:
            equal_var = lev_p > ALPHA
            test_stat, p_value = ttest_ind(g1, g2, equal_var=equal_var)
            test_name = "Independent two-sample t-test"
        else:
            test_stat, p_value = mannwhitneyu(g1, g2, alternative="two-sided")
            test_name = "Mann-Whitney U test"

        st.markdown("---")
        st.markdown(f"### Result: {test_name}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Test statistic", f"{test_stat:.4f}")
        m2.metric("p-value", f"{p_value:.4g}")
        m3.metric("α", f"{ALPHA}")

        if p_value < ALPHA:
            st.success(
                f"**Decision: Reject H₀.** There is a statistically significant difference "
                f"in {metric_col} between {group_names[0]} and {group_names[1]}."
            )
        else:
            st.info(
                f"**Decision: Fail to reject H₀.** There is insufficient evidence that "
                f"{metric_col} differs between {group_names[0]} and {group_names[1]}."
            )

        fig_box = px.box(df, x=factor_col, y=metric_col, color=factor_col, points="all",
                          title=f"{metric_col} by {factor_col}")
        st.plotly_chart(fig_box, use_container_width=True)

    elif len(group_names) >= 3:
        anova_stat, anova_p = f_oneway(*groups.values())
        st.markdown("---")
        st.markdown("### Result: One-Way ANOVA")
        m1, m2, m3 = st.columns(3)
        m1.metric("F-statistic", f"{anova_stat:.4f}")
        m2.metric("p-value", f"{anova_p:.4g}")
        m3.metric("α", f"{ALPHA}")

        if anova_p < ALPHA:
            st.success(
                f"**Decision: Reject H₀.** Mean {metric_col} differs significantly across "
                f"the levels of {factor_col}."
            )
        else:
            st.info(
                f"**Decision: Fail to reject H₀.** Insufficient evidence that mean {metric_col} "
                f"differs across the levels of {factor_col}."
            )

        fig_box = px.box(df, x=factor_col, y=metric_col, color=factor_col, points="all",
                          title=f"{metric_col} by {factor_col}")
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("Selected factor needs at least 2 groups for a comparison test.")

    st.markdown("---")
    st.subheader("Chi-Square Test of Independence (Two Categorical Variables)")
    colC, colD = st.columns(2)
    with colC:
        cat_var_1 = st.selectbox("Categorical variable 1", categorical_columns, index=0, key="chi_1")
    with colD:
        remaining_cats = [c for c in categorical_columns if c != cat_var_1]
        cat_var_2 = st.selectbox("Categorical variable 2", remaining_cats, index=0, key="chi_2")

    contingency = pd.crosstab(df[cat_var_1], df[cat_var_2])
    chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency)

    st.markdown("**Contingency table (observed counts)**")
    st.dataframe(contingency, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Chi-square statistic", f"{chi2_stat:.4f}")
    m2.metric("Degrees of freedom", f"{dof}")
    m3.metric("p-value", f"{chi2_p:.4g}")

    if chi2_p < ALPHA:
        st.success(
            f"**Decision: Reject H₀.** There is a statistically significant association "
            f"between {cat_var_1} and {cat_var_2}."
        )
    else:
        st.info(
            f"**Decision: Fail to reject H₀.** Insufficient evidence of an association "
            f"between {cat_var_1} and {cat_var_2}."
        )

# ============================================================
# TAB 3 — LIVE PREDICTION & DIAGNOSTICS
# ============================================================
with tab3:
    st.subheader("Live Charge Prediction")
    st.write(
        "The model below is the same multiple linear regression fitted in the notebook: "
        f"`charges ~ {' + '.join(feature_columns)}` (fit via `statsmodels.OLS`)."
    )

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            in_age = st.slider("Age", age_min, age_max, int(df["age"].median()))
            in_bmi = st.slider("BMI", round(bmi_min, 1), round(bmi_max, 1), float(round(df["bmi"].median(), 1)))
        with c2:
            in_children = st.slider(
                "Children", int(df["children"].min()), int(df["children"].max()), int(df["children"].median())
            )
            in_sex = st.radio("Sex", sorted(df["sex"].unique()), horizontal=True)
        with c3:
            in_smoker = st.radio("Smoker", sorted(df["smoker"].unique()), horizontal=True)
            in_region = st.selectbox("Region", sorted(df["region"].unique()))
        submitted = st.form_submit_button("Predict charges")

    if submitted:
        input_row = pd.DataFrame([{
            "age": in_age, "bmi": in_bmi, "children": in_children,
            "sex": in_sex, "smoker": in_smoker, "region": in_region,
        }])
        input_dummies = pd.get_dummies(input_row, columns=["sex", "smoker", "region"])
        # align to the training design matrix (missing dummy cols = 0, e.g. baseline category)
        input_aligned = input_dummies.reindex(columns=feature_columns, fill_value=0).astype(float)
        input_aligned = sm.add_constant(input_aligned, has_constant="add")
        input_aligned = input_aligned[X_design.columns]

        pred = ols_model.get_prediction(input_aligned).summary_frame(alpha=ALPHA)

        st.markdown("### Prediction")
        p1, p2, p3 = st.columns(3)
        p1.metric("Predicted charge", f"${pred['mean'].iloc[0]:,.2f}")
        p2.metric("95% CI (mean)", f"${pred['mean_ci_lower'].iloc[0]:,.0f} – ${pred['mean_ci_upper'].iloc[0]:,.0f}")
        p3.metric("95% Prediction interval", f"${pred['obs_ci_lower'].iloc[0]:,.0f} – ${pred['obs_ci_upper'].iloc[0]:,.0f}")
        st.caption(
            "The confidence interval reflects uncertainty in the *average* charge for this "
            "profile; the (wider) prediction interval reflects uncertainty for a *single* "
            "new individual with this profile."
        )

    st.markdown("---")
    st.subheader("Model Summary")
    with st.expander("Coefficients, p-values, 95% CI"):
        coeff_table = pd.DataFrame({
            "Coefficient": ols_model.params,
            "Std. Error": ols_model.bse,
            "P-value": ols_model.pvalues,
            "Lower 95% CI": ols_model.conf_int()[0],
            "Upper 95% CI": ols_model.conf_int()[1],
        }).round(4)
        st.dataframe(coeff_table, use_container_width=True)
        st.write(f"**R²:** {ols_model.rsquared:.4f}  |  **Adjusted R²:** {ols_model.rsquared_adj:.4f}")

    st.markdown("---")
    st.subheader("Residual Diagnostics (Gauss-Markov Checks)")
    fitted_values = ols_model.fittedvalues
    residuals = ols_model.resid

    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        fig_resid = px.scatter(
            x=fitted_values, y=residuals,
            labels={"x": "Fitted Values", "y": "Residuals"},
            title="Residuals vs Fitted Values",
        )
        fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_resid, use_container_width=True)
        st.caption("Look for a random scatter around 0 — no clear funnel or curve — to support linearity & homoscedasticity.")

    with diag_col2:
        qq = stats.probplot(residuals, dist="norm")
        theoretical_q, sample_q = qq[0]
        slope, intercept, _ = qq[1]
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode="markers", name="Residuals"))
        fig_qq.add_trace(go.Scatter(
            x=theoretical_q, y=slope * np.array(theoretical_q) + intercept,
            mode="lines", name="45° reference line", line=dict(color="red", dash="dash"),
        ))
        fig_qq.update_layout(title="Q-Q Plot of Residuals", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        st.plotly_chart(fig_qq, use_container_width=True)
        st.caption("Points close to the reference line support normality of residuals.")

    jb_stat, jb_p = jarque_bera(residuals)
    st.markdown(
        f"**Jarque-Bera test:** statistic = {jb_stat:.2f}, p-value = {jb_p:.4g} → residuals "
        f"{'do not show' if jb_p > ALPHA else 'show'} significant evidence of non-normality "
        f"at α = {ALPHA}."
    )

    st.markdown("**Variance Inflation Factor (multicollinearity check)**")
    X_vif = X_design.drop(columns=["const"])
    vif_data = pd.DataFrame({
        "Feature": X_vif.columns,
        "VIF": [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])],
    })

    def _vif_flag(v):
        if v < 5:
            return "Low multicollinearity"
        elif v < 10:
            return "Moderate multicollinearity"
        return "High multicollinearity"

    vif_data["Interpretation"] = vif_data["VIF"].apply(_vif_flag)
    st.dataframe(vif_data.round(3), use_container_width=True)

st.markdown("---")
st.caption("M.Sc. Data Science · Statistical Modeling with Python · Lab-4 Dashboard")
