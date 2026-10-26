# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Developer**: Nilesh
- **Model Date**: October 2026
- **Model Version**: 1.0.0
- **Model Type**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Hyperparameters**: `n_estimators=100`, `max_depth=15`, `random_state=42`, `n_jobs=-1`

## Intended Use
- **Primary Intended Uses**: This model predicts whether an individual's annual income exceeds $50,000 based on demographic and employment attributes extracted from census data.
- **Primary Intended Users**: Machine learning engineers, researchers, and developers evaluating census-based socio-economic classification tasks or serving inference through a RESTful API.
- **Out-of-Scope Uses**: This model should not be used as the sole basis for credit decisions, loan approvals, employment vetting, or eligibility for public benefits.

## Training Data
- **Source**: Publicly available 1994 US Census Bureau database extracted by Barry Becker (Adult Census dataset).
- **Size**: 80% of the original 32,561 records (approximately 26,048 samples).
- **Features**: Continuous features (such as `age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, and `hours-per-week`) and categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`) encoded using scikit-learn's `OneHotEncoder`.
- **Target**: Binary classification target `salary` (`<=50K` or `>50K`), binarized using `LabelBinarizer`.

## Evaluation Data
- **Source**: 20% held-out test split (6,513 samples) from the census dataset using stratified pseudo-random splitting (`random_state=42`).
- **Data Preprocessing**: Preprocessed identically to training data using the fitted `OneHotEncoder` with `handle_unknown="ignore"`.

## Metrics
The model was evaluated using binary classification metrics focused on positive class identification (`>50K`):
- **Precision**: 0.7918
- **Recall**: 0.5786
- **F1-Score**: 0.6686

Slice-based evaluation across categorical groups (e.g., `workclass`, `education`, `race`, `sex`) was computed and logged to `slice_output.txt` to monitor subgroup disparities.

## Ethical Considerations
- **Demographic Bias**: The underlying 1994 Census dataset contains historical demographic disparities and systemic imbalances in gender, racial categories, and income distribution.
- **Mitigation**: Categorical data slices were evaluated in `slice_output.txt` to track variations in performance metrics across subgroups before deployment.

## Caveats and Recommendations
- **Historical Data**: The data was collected in 1994; monetary thresholds and employment dynamics do not reflect current inflation or contemporary wage distributions.
- **Imbalanced Classes**: High precision (0.7918) alongside moderate recall (0.5786) indicates conservative prediction for the high-income threshold (`>50K`). Users requiring higher recall should tune decision thresholds or apply re-sampling techniques.