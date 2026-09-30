# Car Sales Analytics (Group 9)

End-to-end analysis of a 2,000-row Indian car-sales dataset (43 columns, Jan 2024 to Aug 2026): data
wrangling, EDA, hypothesis tests, **ex-showroom price prediction**, customer/vehicle segmentation,
cancellation and financing classifiers, and factor analysis.

The repo has two parts:

- `notebooks/Final_Project_Group_9.ipynb`: the full analysis (runs in Colab or locally).
- `src/car_sales/` + `scripts/`: the price model packaged for training and prediction from the command line.

## Key results

| Model (log-price target, 20% test set) | R² (INR) | MAE (lakh) | MAPE |
|---|---|---|---|
| Linear Regression | 0.940 | 4.04 | 14.6% |
| Ridge (deployed in `src/`) | 0.940 | 4.06 | 14.7% |
| Lasso (25 of 61 features) | 0.940 | 4.08 | 14.7% |
| Random Forest | 0.933 | 4.17 | 14.8% |
| Decision Tree | 0.924 | 4.31 | 15.0% |

- Vehicle type is by far the strongest price driver: luxury SUVs and sedans price roughly 9x the baseline.
- Simple linear models match the tree ensembles, so the price structure is mostly additive on the log scale.
- K-Means on vehicle and price features gives 7 segments (silhouette 0.635).
- **Cancellation and financing are not predictable from this data** (test AUC 0.548 and 0.488; cross-validated AUC is at or below 0.5). Those classifiers are in the notebook for completeness and should not be deployed.
- Factor analysis is not justified (KMO = 0.30, below the 0.6 threshold).

## Quick start

```bash
git clone https://github.com/<your-username>/car-sales-analytics.git
cd car-sales-analytics
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the dataset at `data/Car_Sales.csv` (it is git-ignored; see `data/README.md`), then:

```bash
# train (80/20 split, seed 42); add --full to refit on all rows afterwards
python scripts/train.py --data data/Car_Sales.csv

# predict for a sample vehicle, or for your own CSV
python scripts/predict.py --example
python scripts/predict.py --input new_cars.csv --output predictions.csv

# run the tests
pytest
```

Input CSV columns for prediction: `Brand, Model, Variant, Vehicle_Type, Fuel_Type, Transmission,
Seating_Capacity, Engine_CC, Mileage_KMPL_or_Range_KM, Manufacturing_Year`.

Use from Python:

```python
from car_sales import load_model, predict_price
model = load_model("models/price_model.joblib")
predict_price(model, new_cars_df)
```

## Project layout

```
notebooks/   full analysis notebook
src/car_sales/   features.py (feature engineering), model.py (train / evaluate / predict)
scripts/     train.py, predict.py
tests/       pytest smoke tests on synthetic data
data/        dataset goes here (not committed)
models/      trained model artefacts (not committed)
```

## Limitations

- Test MAPE is about 15%: fine for segment-level estimates, not for quoting an individual car's price.
- Predictions are `exp(log-price)` with no bias correction, so they run slightly low on average.
- A previously unseen `Model` value is ignored by the encoder, so the prediction falls back to the brand, type and spec effects.
- 566 rows have a manufacturing year later than the sale year; the notebook flags this and keeps them.
- Dataset size is small (2,000 rows), so results may not generalise beyond it.

## License

MIT, see `LICENSE`.
