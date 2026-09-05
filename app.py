import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from data_pipeline import (
    load_everything, CL_FEATURES, bootstrap_prediction_interval, price_band,
)

st.set_page_config(page_title="EduPro — Course Demand & Revenue Forecasting", layout="wide")


@st.cache_resource(show_spinner="Loading data and training models…")
def get_data():
    return load_everything()


data = get_data()
course_perf = data["course_perf"]
models = data["models"]
rollup = data["category_rollup"]
baseline = data["monthly_baseline"]
enrollment_importance = data["enrollment_importance"]
revenue_importance = data["revenue_importance"]

st.title("EduPro — Course Demand & Revenue Forecasting")
st.caption(
    "Course-level lifetime prediction model. Revenue prediction is reliable (CV R² ≈ 0.98); "
    "enrollment-count prediction is shown transparently as exploratory — see the Enrollment tab."
)

tab_overview, tab_revenue, tab_enrollment, tab_importance, tab_category, tab_whatif = st.tabs(
    ["Overview", "Revenue Forecast", "Enrollment (Exploratory)", "Feature Importance", "Category Comparison", "What-If Predictor"]
)

# ------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Courses tracked", len(course_perf))
    col2.metric("Total lifetime revenue", f"${course_perf['TotalRevenue'].sum():,.0f}")
    col3.metric("Total lifetime enrollments", f"{course_perf['TotalEnrollments'].sum():,.0f}")
    col4.metric("Revenue model CV R²", f"{models['revenue_cv']['R2_mean']:.3f}")

    st.markdown("### What this dashboard can and can't tell you")
    c1, c2 = st.columns(2)
    with c1:
        st.success(
            "**Revenue forecasting — reliable.** Course-level lifetime revenue is predictable "
            f"(CV R² ≈ {models['revenue_cv']['R2_mean']:.2f}) mainly because revenue in this dataset "
            "is close to `CoursePrice × a fairly constant enrollment count`. Useful for pricing and "
            "revenue planning."
        )
    with c2:
        st.warning(
            "**Enrollment forecasting — exploratory only.** Every model tested scores a "
            f"**negative** cross-validated R² (best: {models['enrollment_cv']['R2_mean']:.2f}) for "
            "predicting how many people enroll — at monthly, category, or course-lifetime grain. "
            "This is a genuine data limitation, not a modeling gap; treat any enrollment number here "
            "as illustrative, not a forecast."
        )

    st.markdown("### Why there's no month-to-month demand forecast")
    st.write(
        f"Month-to-month enrollment correlation across {baseline['n_months']} months of history is "
        f"**{baseline['autocorrelation']:.3f}** — statistically zero. A naive 'next month = this month' "
        f"baseline already scores R² = {baseline['naive_r2']:.2f}, and no model in the underlying "
        "notebook beat it. More transaction history (24+ months) would be needed before monthly "
        "forecasting is worth attempting again."
    )

# ------------------------------------------------------------------
# REVENUE FORECAST
# ------------------------------------------------------------------
with tab_revenue:
    st.subheader("Predicted vs Actual Lifetime Revenue")
    fig = px.scatter(
        course_perf, x="TotalRevenue", y="PredictedTotalRevenue", hover_name="CourseName",
        color="CourseCategory", labels={"TotalRevenue": "Actual Revenue ($)", "PredictedTotalRevenue": "Predicted Revenue ($)"},
    )
    max_val = max(course_perf["TotalRevenue"].max(), course_perf["PredictedTotalRevenue"].max())
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"CV R² = {models['revenue_cv']['R2_mean']:.3f} · CV MAE = ${models['revenue_cv']['MAE']:,.0f} "
        "(cross-validated — this is the honest out-of-sample estimate, not the in-sample fit above)."
    )

    st.subheader("Top / Bottom Courses by Predicted Revenue")
    n = st.slider("Number of courses to show", 5, 20, 10, key="rev_n")
    c1, c2 = st.columns(2)
    cols_to_show = ["CourseName", "CourseCategory", "CoursePrice", "TotalRevenue", "PredictedTotalRevenue"]
    with c1:
        st.markdown("**Top predicted revenue**")
        st.dataframe(
            course_perf.sort_values("PredictedTotalRevenue", ascending=False)[cols_to_show].head(n),
            width='stretch', hide_index=True,
        )
    with c2:
        st.markdown("**Lowest predicted revenue**")
        st.dataframe(
            course_perf.sort_values("PredictedTotalRevenue", ascending=True)[cols_to_show].head(n),
            width='stretch', hide_index=True,
        )

# ------------------------------------------------------------------
# ENROLLMENT (EXPLORATORY)
# ------------------------------------------------------------------
with tab_enrollment:
    st.warning(
        "**Exploratory tab.** The enrollment model's cross-validated R² is negative "
        f"({models['enrollment_cv']['R2_mean']:.2f}), meaning it performs worse than simply guessing "
        "the average enrollment for every course. Numbers below describe in-sample correlation "
        "patterns only — do not use them for launch/staffing decisions without new data."
    )
    fig = px.scatter(
        course_perf, x="TotalEnrollments", y="PredictedTotalEnrollments", hover_name="CourseName",
        color="CourseCategory", labels={"TotalEnrollments": "Actual Enrollments", "PredictedTotalEnrollments": "Predicted Enrollments"},
    )
    max_val = max(course_perf["TotalEnrollments"].max(), course_perf["PredictedTotalEnrollments"].max())
    min_val = min(course_perf["TotalEnrollments"].min(), course_perf["PredictedTotalEnrollments"].min())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig, width='stretch')
    st.caption(
        f"CV R² = {models['enrollment_cv']['R2_mean']:.3f} · CV MAE = {models['enrollment_cv']['MAE']:.1f} enrollments "
        f"(on a base of ~{course_perf['TotalEnrollments'].mean():.0f} average enrollments per course — "
        "a model this far below zero R² adds no information over the historical average)."
    )

# ------------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------------
with tab_importance:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Revenue model** — trustworthy, since the model generalizes")
        fig = px.bar(revenue_importance, x="Importance", y="Feature", orientation="h")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')
        st.caption("Expect `CoursePrice` to dominate — that's the mechanical relationship described in the Overview tab.")
    with c2:
        st.markdown("**Enrollment model** — descriptive only, model does not generalize")
        fig = px.bar(enrollment_importance, x="Importance", y="Feature", orientation="h")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width='stretch')
        st.caption("Shown for transparency; not a reliable driver list since the underlying model has negative CV R².")

# ------------------------------------------------------------------
# CATEGORY COMPARISON
# ------------------------------------------------------------------
with tab_category:
    st.subheader("Predicted vs Actual Revenue by Category")
    fig = go.Figure()
    fig.add_bar(name="Actual", x=rollup["CourseCategory"], y=rollup["ActualRevenue"])
    fig.add_bar(name="Predicted", x=rollup["CourseCategory"], y=rollup["PredictedRevenue"])
    fig.update_layout(barmode="group", xaxis_tickangle=-30)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Category Summary")
    st.dataframe(rollup, width='stretch', hide_index=True)
    st.caption(
        "Rolled up from the course-level revenue model (reliable). Category-level *monthly* revenue "
        "was also tested in the notebook and looked predictable (R² ≈ 0.88), but that signal turned "
        "out to be structural — driven by fixed course pricing composition per category, not a "
        "genuine month-to-month trend — so it's not reproduced here as a timeline."
    )

# ------------------------------------------------------------------
# WHAT-IF PREDICTOR
# ------------------------------------------------------------------
with tab_whatif:
    st.subheader("Predict performance for a new or hypothetical course")
    st.caption("Revenue estimate is reliable. Enrollment estimate is shown for completeness only — treat it as a rough illustration, not a forecast.")

    categories = sorted(course_perf["CourseCategory"].dropna().unique())
    types = sorted(course_perf["CourseType"].dropna().unique())
    levels = sorted(course_perf["CourseLevel"].dropna().unique())

    c1, c2, c3 = st.columns(3)
    with c1:
        category = st.selectbox("Course Category", categories)
        course_type = st.selectbox("Course Type", types)
        level = st.selectbox("Course Level", levels)
    with c2:
        price = st.slider("Course Price ($)", 0, 500, 100, step=5)
        duration = st.slider("Course Duration (hours)", 1, int(course_perf["CourseDuration"].max()) + 5, 20)
        rating = st.slider("Course Rating", 1.0, 5.0, 4.0, step=0.1)
    with c3:
        experience = st.slider("Instructor Years of Experience", 0, int(course_perf["YearsOfExperience"].max()) + 2, 5)
        teacher_rating = st.slider("Instructor Rating", 1.0, 5.0, 4.0, step=0.1)

    query = pd.DataFrame([{
        "CourseCategory": category, "CourseType": course_type, "CourseLevel": level,
        "CoursePrice": price, "CourseDuration": duration, "CourseRating": rating,
        "YearsOfExperience": experience, "TeacherRating": teacher_rating,
        "PriceBand": price_band(price),
        "DurationBucket": pd.cut([duration], bins=[-np.inf, course_perf["CourseDuration"].quantile(0.33), course_perf["CourseDuration"].quantile(0.66), np.inf], labels=["Short", "Medium", "Long"])[0],
        "RatingTier": pd.cut([rating], bins=[-np.inf, 3.5, 4.2, np.inf], labels=["Low", "Medium", "High"])[0],
        "ExperienceBucket": pd.cut([experience], bins=[-np.inf, 3, 7, np.inf], labels=["Junior", "Mid", "Senior"])[0],
    }])[CL_FEATURES]

    if st.button("Predict", type="primary"):
        with st.spinner("Running bootstrap intervals…"):
            rev_point, rev_lo, rev_hi = bootstrap_prediction_interval(
                models["revenue_model"], course_perf[CL_FEATURES], course_perf["TotalRevenue"], query, n_boot=100
            )
            enr_point, enr_lo, enr_hi = bootstrap_prediction_interval(
                models["enrollment_model"], course_perf[CL_FEATURES], course_perf["TotalEnrollments"], query, n_boot=100
            )
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted lifetime revenue", f"${rev_point[0]:,.0f}", help="90% bootstrap interval shown below")
            st.caption(f"Range: ${rev_lo[0]:,.0f} – ${rev_hi[0]:,.0f}")
        with c2:
            st.metric("Predicted lifetime enrollments (exploratory)", f"{enr_point[0]:,.0f}")
            st.caption(
                f"Range: {enr_lo[0]:,.0f} – {enr_hi[0]:,.0f} · "
                "⚠️ model has negative CV R² — this interval reflects training variance only, "
                "not real predictive uncertainty. Do not use for planning."
            )

st.divider()
st.caption(
    "Model details: course-level lifetime revenue via Gradient Boosting (tuned), enrollment via Lasso (tuned). "
    "Both exclude CourseID as a feature to avoid memorizing individual courses. See the accompanying notebook "
    "for the full methodology, monthly/category analyses, and honest evaluation."
)
