# GROUP02_CENG313
Jane Street Real-Time Market Data Forecasting

## Project Overview

- **Course:** CENG313 - Data Science
- **Kaggle Competition:** [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- **GitHub Repository:** `GROUP02_CENG313`
- **GitHub Organization:** [ceng313-2026](https://github.com/ceng313-2026)

This project focuses on predicting financial market responses (`responder_6`) using historical market data provided by Jane Street. Due to the dataset's scale, high noise-to-signal ratio and non-stationary nature of financial time-series data, the project uses memory-conscious data inspection and preprocessing, chronological validation, and machine-learning regression models.

## Team Members

| Name Surname | Student ID | GitHub Username |
|---|---:|---|
| Rümeysa Demirelli | 24118080023 | [@godsuyasan](https://github.com/godsuyasan) |
| Emine Yılmaz | 23118080050 | [@emiyilm](https://github.com/emiyilm) |
| Hatice Kübra Küskü | 24118080403 | [@hkubra00](https://github.com/hkubra00) |

## Kaggle Competition and Dataset

The selected competition is [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting). The dataset contains multi-partitioned Apache Parquet files (`train.parquet`, `test.parquet`, and `lags.parquet`) and metadata CSV files (`features.csv` and `responders.csv`).

### Dataset Summary

- **Total rows audited:** 47,127,338 observations across 10 Hive partitions (`partition_id=0..9`, about 11.5 GB on disk).
- **Temporal span:** 1,699 unique trading days (`date_id` from `0` to `1698`) and 968 intraday time steps (`time_id` from `0` to `967`), roughly 27,700 rows per day.
- **Assets:** 39 unique financial instruments (`symbol_id`).
- **Features:** 79 anonymized feature columns:
  - 76 continuous features (`Float32`).
  - 3 discrete integer features (`feature_09`, `feature_10`, and `feature_11`).
- **Targets and weights:** 9 responder columns (`responder_0` to `responder_8`). The primary competition target is `responder_6`; observations are evaluated using the provided `weight` column.
- **Lags:** `lags.parquet` holds the previous day's value of each responder (`responder_k_lag_1`).

> **Data policy:** Dataset files are not included in this repository. Competition data is downloaded from Kaggle by the notebook (into `jane_street_data/`, which is excluded through `.gitignore`), so Kaggle credentials must be configured when running it.

## Assignment 1: EDA and Baseline Implementations

The notebook `assignment1_eda_baseline.ipynb` follows the required sections: (1) Problem definition, (2) Data overview, (3) Exploratory data analysis, (4) Preprocessing, (5) Baseline model, (6) Kaggle submission and (7) Conclusion.

### Data Overview

- **Duplicates:** No repeated (`date_id`, `time_id`, `symbol_id`) keys were found in partition 0.
- **Summary statistics:** `describe()` on a systematic sample of every 100th row (about 471,000 rows) shows that most features have a standard deviation close to 1, but some are not centered (for example `feature_00`, `feature_02` and `feature_03` have means around 0.57). Several features have extreme outliers (for example `feature_05` to `feature_07` reach about -24 to +31, and `feature_14` reaches 119). The three integer features behave like category codes rather than measurements.
- **Missing-value audit:** Exact null counts were extracted from Parquet row-group metadata using `pyarrow.parquet`, avoiding the need to load the complete dataset into memory. Only features contain missing values.
  - 32 features had no missing values.
  - 34 features had a missing-value ratio between 0% and 5%.
  - 13 features had a missing-value ratio between 5% and 20%.
  - No features had more than 20% missing values. The highest reported missingness was 17.90% (`feature_21`, `26`, `27`, `31`).
  - Features go missing in groups with identical null counts, which suggests they come from the same source.

### Exploratory Data Analysis

- **Target and weight analysis:** `responder_6` is centered near zero (mean -0.0021, median -0.025, std 0.89) with heavy tails (skewness 0.47, kurtosis 6.86). It is clipped to [-5, 5]: 49,535 rows sit at -5 and 107,216 rows at +5. `weight` has mean 2.01, maximum 10.24 and no zero or negative values.
- **Distribution shift:** Scaler statistics from the training days used for the baselines (dates 851–1350) differ from the sample statistics over all days (for example `feature_00` has a mean of about 0.25 in the training range and about 0.57 in the global sample), which suggests the feature distributions change over time.

### Validation Strategy for the Closed Competition

The Jane Street competition is treated as a closed competition for this course project. Since a standard Kaggle late submission is not available to us, we evaluate the baseline models locally using a chronological **purged held-out split**, in line with the course instructions. A time-based split is used instead of a random one because a random split would leak future days into the training set.

The dataset contains 1,699 trading days (`date_id` 0–1698). The final chronological block is reserved for evaluation (343 of 1,699 days, about 20%):

| Partition | Trading days (`date_id`) | Purpose |
|---|---:|---|
| Training period | 851–1350 (500 days) | Data available for fitting the baseline models |
| Purge gap | 1351–1355 (5 days) | Excluded to reduce temporal leakage between training and evaluation |
| Held-out evaluation period | 1356–1698 (343 days) | Reserved for local validation/evaluation |

For the baseline experiments, due to Google Colab memory constraints, the notebook uses a sample of 500,000 training observations and 100,000 held-out observations from the corresponding periods.

In addition to the final held-out evaluation period, a four-fold chronological sliding-window scheme was designed with a maximum training window of 500 days and a 5-day purge gap:

- **Fold 1:** Train `0–333`; purge gap; validation `339–677`.
- **Fold 2:** Train `173–672`; purge gap; validation `678–1016`.
- **Fold 3:** Train `512–1011`; purge gap; validation `1017–1355`.
- **Fold 4:** Train `851–1350`; purge gap; held-out evaluation `1356–1698`.

Only Fold 4 has been run for the baselines so far.

### Preprocessing

- Training means and standard deviations are calculated in batches on the training range only, to avoid leakage.
- Extreme standardized values are clipped to the range `[-10.0, +10.0]`.
- Missing values are imputed with `0.0` in standardized (Z-score) space.
- Rows with `weight <= 0` are dropped, and the three discrete features are cast to float.
- Preprocessed data is streamed to Parquet using Polars `sink_parquet` to reduce memory usage.

### Reusable Code

Reusable classes and helper functions are maintained in the `src/` directory and imported by the notebooks:

- `src/data.py`: chronological data splitting and preprocessing utilities.
- `src/models.py`: baseline model classes (Zero, Ridge, LightGBM and CatBoost wrappers).
- `src/utils.py`: the competition's weighted R² metric and memory-conscious Parquet loading utilities.

### Baseline Models and Results

The baseline models are evaluated using the competition's weighted zero-mean R² metric:

$$
R^2_{\mathrm{weighted}} = 1 - \frac{\sum_i w_i (y_i-\hat{y}_i)^2}{\sum_i w_i y_i^2}
$$

The reported experiments use 500,000 training observations and 100,000 held-out validation observations from Fold 4.

| Model | Train Weighted R² | Validation Weighted R² | Training and inference time | Note |
|---|---:|---:|---:|---|
| Zero Baseline | 0.000000 | 0.000000 | 0.00 s | Predicts zero for every observation. |
| Ridge Regression (`alpha=100`) | 0.009488 | -0.005350 | 0.81 s | Positive training score, negative held-out score. |
| LightGBM Regressor | 0.033057 | **0.002207** | 18.78 s | Best held-out score among the tested baselines. |
| CatBoost Regressor | 0.006927 | 0.000774 | 9.49 s | Small positive held-out score. |

**Main observations:**

1. The signal is very weak. Ridge Regression has a positive training score but a negative validation score, indicating limited generalization in this temporal setting.
2. LightGBM achieves the best held-out Weighted R² (`0.002207`), but its training score is much higher (`0.033057`), so it fits the training period and loses most of that on later days. This is consistent with the distribution shift described above.
3. CatBoost is slightly positive (`0.000774`). The gaps between the models are tiny and come from one split with 100,000 validation rows, so we do not claim a reliable ranking between LightGBM and CatBoost.
4. The LightGBM split-importance analysis identifies `feature_06`, `feature_07`, `feature_04`, `feature_60`, `feature_05`, and `feature_36` as the most important features in the reported experiment.

**Limitations:** only one of the four folds was evaluated, only 500,000 of the roughly 14M training-range rows were used, and the baselines do not use `lags.parquet` or the other responders.

**Next steps:** add lag features from `lags.parquet`, evaluate on all four folds with more rows, handle outliers more carefully and treat `feature_09`, `feature_10` and `feature_11` as categories, add per-symbol and time-of-day features, and tune LightGBM with stronger regularization.

## Assignment 2: Proposed Solution

This section will be updated when the proposed solution has been implemented and evaluated. It will describe the proposed method, its motivation, implementation details, validation setup and results.

## Current Repository Structure

```text
GROUP02_CENG313/
├── README.md
├── requirements.txt
├── .gitignore
├── assignment1_eda_baseline.ipynb
├── assignment2_solution.ipynb        (to be added for Assignment 2)
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── utils.py
├── data/                             (not committed)
└── submissions/                      (submission score screenshots)
```

The repository must not contain downloaded competition data or other large data files. Reusable code belongs in `src/`, rather than being duplicated only inside notebook cells.

## How to Run the Notebooks

### Assignment 1

1. Open `assignment1_eda_baseline.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Configure Kaggle credentials in Colab Secrets using the names expected by the notebook (`KAGGLE_USERNAME` and `KAGGLE_KEY`).
3. Ensure the required Python packages are installed (see `requirements.txt`). The notebook includes the relevant `pip install` steps.
4. Run the notebook cells in order. The notebook audits the data, performs preprocessing, imports reusable components from `src/` and runs the baseline experiments.

A T4 GPU runtime should be selected in Colab. The baseline implementations should be run using the runtime configuration documented in the notebook.

## Progress Logs

- **October 2026:** Formed Group 02, selected the Jane Street Real-Time Market Data Forecasting competition and initialized the project repository.
- **November 10, 2026:** Implemented Kaggle data access, inspected the dataset schema with Polars and performed a memory-conscious missing-value audit using PyArrow metadata.
- **November 11, 2026:** Analyzed the `responder_6` and `weight` distributions, designed chronological purged splits and prepared the Fold 4 data.
- **November 12–13, 2026:** Implemented reusable utilities and baseline model components; evaluated Zero, Ridge, LightGBM and CatBoost baselines; documented the held-out validation strategy for the closed competition.
- **November 13, 2026:** Wrote the data overview and conclusion, organized the notebook into the seven required sections and updated the README.

## Kaggle Submission History

No Kaggle submission is reported for Assignment 1.

**Competition closed — evaluated on a held-out split.** The baseline results are reported using the local purged held-out evaluation strategy described above in "**Validation Strategy for the Closed Competition**". The best held-out weighted R² is `0.002207` (LightGBM).
