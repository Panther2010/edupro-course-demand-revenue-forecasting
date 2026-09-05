"""
EduPro dashboard data & modeling pipeline.

Mirrors the notebook's course-level lifetime model (Sections 6-11):
- Revenue model: reliable, R^2 ~0.98 CV (mostly price-driven — see notebook for the honest caveat)
- Enrollment model: exploratory only — every model scores negative CV R^2 in the notebook.
  Kept here for completeness / transparency, never presented as a reliable forecast.

Pure functions, no Streamlit imports, so this can be unit-tested / run standalone.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error

RANDOM_STATE = 42

CL_FEATURES = [
    "CourseCategory", "CourseType", "CourseLevel",
    "CoursePrice", "CourseDuration", "CourseRating",
    "YearsOfExperience", "TeacherRating",
    "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket",
]
CL_CATEGORICAL = ["CourseCategory", "CourseType", "CourseLevel", "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket"]
CL_NUMERICAL = [c for c in CL_FEATURES if c not in CL_CATEGORICAL]

# Tuned hyperparameters carried over from the notebook's Section 8 GridSearchCV
ENROLLMENT_MODEL = RandomForestRegressor(
    max_depth=3, min_samples_leaf=1, n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
)
REVENUE_MODEL = Lasso(alpha=1.0, max_iter=10000)


def resolve_data_path():
    candidates = [
        "data/EduPro_Online_Platform.xlsx",
        "EduPro_Online_Platform.xlsx",
        "../data/EduPro_Online_Platform.xlsx",
    ]
    return next((p for p in candidates if os.path.exists(p)), candidates[0])


def price_band(p):
    # Calibrated to this dataset's actual paid-course range ($0.78-$490.90),
    # not an arbitrary $0-60 scale.
    if p == 0:
        return "Free"
    if p <= 150:
        return "Low"
    if p <= 300:
        return "Medium"
    return "High"


def load_raw(file_path=None):
    file_path = file_path or resolve_data_path()
    xl = pd.ExcelFile(file_path)
    users = pd.read_excel(xl, sheet_name="Users")
    teachers = pd.read_excel(xl, sheet_name="Teachers")
    courses = pd.read_excel(xl, sheet_name="Courses")
    transactions = pd.read_excel(xl, sheet_name="Transactions")
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"], errors="coerce")
    transactions["MonthStart"] = transactions["TransactionDate"].dt.to_period("M").dt.to_timestamp()
    return users, teachers, courses, transactions


def build_course_perf(courses, teachers, transactions):
    """Course-level lifetime table: one row per course, static features + realized totals."""
    course_teacher = (
        transactions.groupby(["CourseID", "TeacherID"]).size().reset_index(name="n")
        .sort_values(["CourseID", "n"], ascending=[True, False])
    )
    primary_teacher = course_teacher.drop_duplicates("CourseID")[["CourseID", "TeacherID"]]

    course_perf = (
        transactions.groupby("CourseID")
        .agg(TotalEnrollments=("TransactionID", "count"), TotalRevenue=("Amount", "sum"))
        .reset_index()
        .merge(courses, on="CourseID", how="left")
        .merge(primary_teacher, on="CourseID", how="left")
        .merge(teachers, on="TeacherID", how="left")
    )
    course_perf["RevenuePerEnrollment"] = course_perf["TotalRevenue"] / course_perf["TotalEnrollments"]
    course_perf["PriceBand"] = course_perf["CoursePrice"].apply(price_band)
    course_perf["DurationBucket"] = pd.qcut(course_perf["CourseDuration"], q=3, labels=["Short", "Medium", "Long"], duplicates="drop")
    course_perf["RatingTier"] = pd.cut(course_perf["CourseRating"], bins=[-np.inf, 3.5, 4.2, np.inf], labels=["Low", "Medium", "High"])
    course_perf["ExperienceBucket"] = pd.cut(course_perf["YearsOfExperience"], bins=[-np.inf, 3, 7, np.inf], labels=["Junior", "Mid", "Senior"])
    return course_perf


def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), CL_NUMERICAL),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CL_CATEGORICAL),
    ])


def cv_scores(pipe, X, y, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
    cv = cross_validate(pipe, X, y, cv=kfold, scoring=scoring)
    return {
        "MAE": -cv["test_MAE"].mean(),
        "RMSE": -cv["test_RMSE"].mean(),
        "R2_mean": cv["test_R2"].mean(),
        "R2_std": cv["test_R2"].std(),
    }


def train_models(course_perf):
    """Fit both course-level models on the full table and return fitted pipelines + CV metrics."""
    enrollment_pipe = Pipeline([("prep", build_preprocessor()), ("model", clone(ENROLLMENT_MODEL))])
    revenue_pipe = Pipeline([("prep", build_preprocessor()), ("model", clone(REVENUE_MODEL))])

    X = course_perf[CL_FEATURES]
    enrollment_metrics = cv_scores(enrollment_pipe, X, course_perf["TotalEnrollments"])
    revenue_metrics = cv_scores(revenue_pipe, X, course_perf["TotalRevenue"])

    enrollment_pipe.fit(X, course_perf["TotalEnrollments"])
    revenue_pipe.fit(X, course_perf["TotalRevenue"])

    return {
        "enrollment_model": enrollment_pipe,
        "revenue_model": revenue_pipe,
        "enrollment_cv": enrollment_metrics,
        "revenue_cv": revenue_metrics,
    }


def bootstrap_prediction_interval(model, X, y, X_query, n_boot=150, alpha=0.1, random_state=RANDOM_STATE):
    """80/90%-style interval via bootstrap resampling of the training set. Returns (point, lower, upper)
    for each row of X_query. NOTE: reflects training variance, not out-of-sample generalization error —
    see enrollment-model caveat in the app."""
    rng = np.random.RandomState(random_state)
    n = len(X)
    boot_preds = np.zeros((n_boot, len(X_query)))
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        m = clone(model)
        m.fit(X.iloc[idx], y.iloc[idx])
        boot_preds[b] = np.clip(m.predict(X_query), 0, None)
    point = np.clip(model.predict(X_query), 0, None)
    lower = np.percentile(boot_preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(boot_preds, 100 * (1 - alpha / 2), axis=0)
    return point, lower, upper


def permutation_importance_df(model, X, y, n_repeats=30):
    perm = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE, scoring="r2")
    return pd.DataFrame({"Feature": X.columns, "Importance": perm.importances_mean}).sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)


def category_rollup(course_perf, models):
    X = course_perf[CL_FEATURES]
    course_perf = course_perf.copy()
    course_perf["PredictedTotalEnrollments"] = np.clip(models["enrollment_model"].predict(X), 0, None)
    course_perf["PredictedTotalRevenue"] = np.clip(models["revenue_model"].predict(X), 0, None)
    return (
        course_perf.groupby("CourseCategory")
        .agg(
            NumCourses=("CourseID", "count"),
            ActualEnrollments=("TotalEnrollments", "sum"),
            PredictedEnrollments=("PredictedTotalEnrollments", "sum"),
            ActualRevenue=("TotalRevenue", "sum"),
            PredictedRevenue=("PredictedTotalRevenue", "sum"),
        )
        .reset_index()
        .sort_values("PredictedRevenue", ascending=False)
    ), course_perf


def monthly_naive_baseline_stats(courses, transactions):
    """Reproduces the notebook's Section 4 finding: monthly per-course enrollment is not
    predictable — used to power the 'why no monthly forecast' panel in the app."""
    all_courses = courses["CourseID"].unique()
    all_months = pd.date_range(transactions["MonthStart"].min(), transactions["MonthStart"].max(), freq="MS")
    grid = pd.MultiIndex.from_product([all_courses, all_months], names=["CourseID", "MonthStart"]).to_frame(index=False)
    course_monthly = (
        transactions.groupby(["CourseID", "MonthStart"]).size().reset_index(name="EnrollmentCount")
    )
    course_monthly = grid.merge(course_monthly, on=["CourseID", "MonthStart"], how="left").fillna(0)
    course_monthly = course_monthly.sort_values(["CourseID", "MonthStart"])
    course_monthly["PrevMonth"] = course_monthly.groupby("CourseID")["EnrollmentCount"].shift(1)
    valid = course_monthly.dropna()
    corr = valid["PrevMonth"].corr(valid["EnrollmentCount"])
    baseline_mae = mean_absolute_error(valid["EnrollmentCount"], valid["PrevMonth"])
    baseline_r2 = r2_score(valid["EnrollmentCount"], valid["PrevMonth"])
    return {"autocorrelation": corr, "naive_mae": baseline_mae, "naive_r2": baseline_r2, "n_months": len(all_months)}


def load_everything(file_path=None):
    """One-call entry point used by the Streamlit app (wrapped in st.cache_resource there)."""
    users, teachers, courses, transactions = load_raw(file_path)
    course_perf = build_course_perf(courses, teachers, transactions)
    models = train_models(course_perf)
    rollup, course_perf_with_preds = category_rollup(course_perf, models)
    baseline = monthly_naive_baseline_stats(courses, transactions)
    enrollment_importance = permutation_importance_df(models["enrollment_model"], course_perf[CL_FEATURES], course_perf["TotalEnrollments"])
    revenue_importance = permutation_importance_df(models["revenue_model"], course_perf[CL_FEATURES], course_perf["TotalRevenue"])
    return {
        "courses": courses,
        "teachers": teachers,
        "transactions": transactions,
        "course_perf": course_perf_with_preds,
        "models": models,
        "category_rollup": rollup,
        "monthly_baseline": baseline,
        "enrollment_importance": enrollment_importance,
        "revenue_importance": revenue_importance,
    }


if __name__ == "__main__":
    data = load_everything()
    print("Course-level rows:", len(data["course_perf"]))
    print("Enrollment CV:", data["models"]["enrollment_cv"])
    print("Revenue CV:", data["models"]["revenue_cv"])
    print("Monthly naive baseline:", data["monthly_baseline"])
    print(data["category_rollup"].head())
