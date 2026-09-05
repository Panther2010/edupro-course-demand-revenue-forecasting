# EduPro — Course Demand & Revenue Forecasting

Predictive modeling project for an online learning platform (EduPro), covering exploratory data analysis, model development, and an interactive dashboard for two business questions:

1. **How much revenue will a course generate?** — reliably predictable (CV R² ≈ 0.97–0.98)
2. **How many people will enroll?** — not predictable from the available data, at any grain tested (monthly, category, or course-lifetime), and this repo documents *why* rather than hiding it

## TL;DR

- Revenue prediction works well because revenue in this dataset is close to `CoursePrice × a fairly constant enrollment count` — genuinely useful for pricing/planning.
- Enrollment prediction was tested six ways (Linear, Ridge, Lasso, Poisson, Random Forest, Gradient Boosting) at three levels of granularity, benchmarked against naive baselines, hyperparameter-tuned, and stress-tested across 20 randomized cross-validation splits. Every version scored negative R². Diagnostics point to the enrollment signal simply not being present in the available features (courses have no stable single instructor; learner demographics barely vary by course).
- Full evidence trail, including the R² instability analysis, is in the notebook and the accompanying report.

## Repo structure

```
.
├── EduPro_Predictive_Modeling.ipynb      # Full analysis: EDA → modeling → tuning → evaluation
├── dashboard/
│   ├── app.py                            # Streamlit UI (6 tabs)
│   ├── data_pipeline.py                  # Data loading, feature engineering, model training (no Streamlit dependency)
│   ├── requirements.txt
│   ├── README.md                         # Dashboard-specific run instructions
│   └── data/
│       └── EduPro_Online_Platform.xlsx
├── reports/
│   ├── EduPro_EDA_Insights_Recommendations.docx
│   └── EduPro_Executive_Summary.docx
└── README.md                             # This file
```

*(Adjust the tree above if your local layout differs — e.g. if `reports/` files currently sit at the repo root.)*

## Running the dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dataset is bundled under `dashboard/data/`. If you move it, update `resolve_data_path()` in `data_pipeline.py`.

## Running the notebook

Requires `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`. Open `EduPro_Predictive_Modeling.ipynb` and run all cells top to bottom — it's a single linear pipeline (no hidden state between out-of-order cells).

## Key modeling notes

- `CourseID` is deliberately excluded as a model feature everywhere — with only 60 courses, one-hot encoding it would let the model memorize individual courses instead of generalizing.
- All reported R² values are cross-validated, not in-sample. Where relevant, both are shown side by side to make overfitting visible.
- Enrollment R² is reported as a *distribution* (per-fold, and across repeated randomized splits), not a single number — at n=60 a single split can be misleading in either direction.

## Documents

- **`reports/EduPro_EDA_Insights_Recommendations.docx`** — full technical writeup: EDA, both models, the R² instability evidence, category-level findings, and recommendations.
- **`reports/EduPro_Executive_Summary.docx`** — 3-page, non-technical summary for stakeholders.

## License

MIT — see `LICENSE`.
