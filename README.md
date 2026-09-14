# House Prices Predictive Modeling Pipeline

A clean, robust machine learning pipeline designed to predict home values using regularized linear regression. This implementation modernizes a baseline preprocessing workflow by eliminating data leakage, correcting domain data types, and stabilizing environmental library dependencies.

##  Key Improvements & Engineering Fixes

* **Zero Data Leakage:** Imputation parameters (like numerical medians) are computed strictly on the training partition (`X_train`) and carried forward safely.
* ** Structural Alignment:** Built using `scikit-learn` Pipelines and `ColumnTransformer` to handle one-hot matrices cleanly, avoiding raw array concatenation index mismatches.
* **Domain Feature Typing:** Explicitly identifies and handles variables like `MSSubClass` (numerical codes representing dwelling types) as unordered categorical factors.
* **Null Value Handling:** Reclassifies structural missing records (where `NA` explicitly stands for a missing asset feature like "No Pool") into static string classes rather than erroring out.
* **Local Package Protection:** Includes environmental configuration guards to bypass broken global system caches (such as conflicting `Pillow`/`PIL` extensions) by forcing local user package paths first.

##  Pipeline Architecture

1. **Ingestion & Correction:** Reads raw dataset fields and explicitly casts structural categorical values to strings.
2. **Train-Test Isolation:** Segregates a clean 20% evaluation dataset split before any imputation matrix operations occur.
3. **Numeric Pipeline:** Imputes missing figures via localized structural medians.
4. **Categorical Pipeline:** Groups nominal string values, patches missing tags with a static `"None"` value fallback, and transforms features into dense categorical matrices using `OneHotEncoder`.
5. **Estimator Tuning:** Fits regularized structural weights onto independent fields using a linear `Ridge` regression module.

## Diagnostic Visualizations Included

* **Residual Scatter Map:** Tracks overall predictive variances to locate data bias trends.
* **Feature Weights Bar Graph:** Isolates the top 15 structural coefficients to rank which property factors influence overall real estate values the most.

##Getting Started

### Prerequisites
Install all relevant dependency configurations cleanly inside your local workspace profile:
```bash
pip install pandas numpy scikit-learn matplotlib pillow --user
```

### Execution Flow
1. Execute the **Initialization & Profile Guard Block** to securely load internal data libraries and isolate feature frameworks.
2. Run the **Transformation & Regression Engine Block** to execute modeling pipelines, print validation error boundaries (`MAE` / `R²`), and visualize diagnostics maps.
