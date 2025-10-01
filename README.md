# Forecasting-Crop-Yield-Using-a-Multilayer-Perceptron-PyTorch-
Predicting next year’s agricultural yield from environmental, land-cover, and geospatial signals using a three-layer MLP in PyTorch. The pipeline includes temporal target shifting, strict time-based splits, careful preprocessing, and quantitative evaluation (RMSE/MAE/R²).

1) Project Overview
	•	Goal: Forecast next-year crop yield at country–crop level using the previous year’s conditions.
	•	Approach: Build a Multilayer Perceptron (MLP) with PyTorch; engineer features from environmental time series and land-cover data; create a target_yield by lagging one year; enforce chronological train/validation split.
	•	Key Result (validation on 2021 → predict 2022):
	•	RMSE: ~7,290
	•	MAE: ~5,504
	•	R²: ~0.770
2) Data & Target Construction
	•	Final merged dataset ≈ 62,498 rows × 36 columns across 2010–2022.
	•	Target (target_yield): created by pairing features from year t with yield from year t+1.
	•	Train/Validation split:
	•	Train: 2010–2020
	•	Validation: 2021 (used to forecast 2022, then compared with actual 2022).
	•	Granularity: Country–crop–year.
	•	Geospatial mapping: Environmental data aggregated annually and mapped to countries via nearest-centroid.
	•	Land cover: 17 land-cover percentage features (e.g., forests, shrublands, croplands, urban, wetlands, snow/ice, water).
	•	Categorical: country, Item (crop), one-hot encoded.
3) Features (Examples)
Environmental (annual averages):
	•	Rainfall, snowfall, terrestrial water storage
	•	Soil moisture (0–10, 10–40, 40–100, 100–200 cm)
	•	Soil temperature (same depths as above)
	•	Plant canopy surface water
Land Cover (percentages):
	•	17 classes: forests (evergreen/deciduous), shrublands, savannas, croplands, urban, wetlands, snow/ice, water
Categorical:
	•	Country, crop (OHE)
4) Preprocessing
	•	Filtering: kept only yield rows; removed non-essential metadata.
	•	Aggregation: monthly → annual at (lat, lon, year), then aggregated to (country, year).
	•	Target shifting: built target_yield as yield_(t+1).
	•	Duplicates/missing: dropped duplicates and NaNs.
	•	Outliers: removed values above 95th percentile for selected columns.
	•	Scaling: StandardScaler applied to numeric + OHE features; same scaler used to inverse-transform predictions.
5) Model
	•	Architecture (PyTorch):
	•	Input: preprocessed feature vector
	•	Hidden: 2 fully connected layers with ReLU ([64, 128] chosen)
	•	Output: 1 neuron (regression)
	•	Loss/Optimizer: MSE loss, Adam optimizer
	•	Batching: DataLoader with shuffled training batches (size 32)
	•	Early stopping: best model saved based on validation loss
6) Training & Hyperparameters
	•	Batch size: tried 16, 32, 64 → final choice 32
	•	Hidden layers: tried [32,64] and [64,128] → final choice [64,128]
	•	Learning rate: tried 1e-5, 1e-4, 5e-4 → final choice 1e-4
	•	Epochs: tested 100, 150, 200 → final choice 150
7) Evaluation
	•	Validation set (predicting 2022):
	•	RMSE: ~7,290.6
	•	MAE: ~5,504.2
	•	R²: ~0.770
	•	Diagnostic: predictions clustered tightly around actual values; wider variance at very high yields.
	•	Observation: time-based split avoids leakage and mimics real forecasting — training only on past, scoring on unseen future.


