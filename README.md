# ML Capstone Studio

**Machine Learning Capstone Project — DATA MINDS**

An interactive machine-learning application that brings together three ML tasks in one Streamlit interface:

- **Regression:** Crop-yield prediction
- **Classification:** Facial-emotion recognition using FER2013
- **Clustering:** Earthquake-zone analysis

The project includes Jupyter notebooks for the experiments and a Streamlit GUI (`app.py`) for exploring the models, predictions, visualizations, and downloadable results.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Setup and Run](#setup-and-run)
- [How to Use the Application](#how-to-use-the-application)
- [Machine-Learning Methods and Evaluation](#machine-learning-methods-and-evaluation)
- [Notebooks](#notebooks)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

## Project Overview

The goal of this capstone is to demonstrate three common machine-learning problem types using real datasets:

| Problem type | Application | Main goal |
|---|---|---|
| Regression | Crop-yield prediction | Estimate crop yield from agricultural and environmental features |
| Classification | Facial-emotion recognition | Predict an emotion label from a face image |
| Clustering | Earthquake-zone analysis | Group earthquake events by geographic location and depth |

The Streamlit application provides a point-and-click interface so users can run analyses without manually executing every notebook cell.

## Project Structure

```text
ML-Capstone-Project-main/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── crop_yield/
│   │   ├── pesticides.csv
│   │   ├── rainfall.csv
│   │   ├── temp.csv
│   │   ├── yield.csv
│   │   └── yield_df.csv
│   └── earthquake/
│       └── database.csv
└── notebooks/
    ├── 01_Classification_Emotion.ipynb
    ├── 02_Regression_Crop_Yield.ipynb
    └── 03_Clustering_Earthquakes_Corrected.ipynb
```

`ML-Capstone-Project-with-GUI.zip`, if present in your copy, is an additional archive included with the project.

## Features

### 1. Crop-Yield Prediction — Regression

- Uses the bundled crop-yield dataset or a user-provided CSV.
- Uses crop and country/area information alongside year, rainfall, pesticide usage, and average temperature.
- Creates a rainfall–pesticide interaction feature.
- Compares several regression algorithms.
- Reports evaluation metrics such as **R²**, **RMSE**, and **MAE**.
- Provides prediction and data-exploration workflows, with downloadable results where available.

**Expected target column:** `hg/ha_yield`

The bundled dataset should include the columns used by the application, including `Area`, `Item`, `Year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp`, and `hg/ha_yield`.

### 2. Facial-Emotion Recognition — Classification

- Supports training from a user-supplied `fer2013.csv` dataset or loading a previously saved model.
- Supports facial-image input through the GUI.
- Uses 48 × 48 grayscale face images and a seven-class emotion setup.
- Provides classifier training and classification evaluation.

**FER2013 CSV format:** the file should contain `emotion` and `pixels` columns. The `pixels` field normally contains the grayscale pixel values for a 48 × 48 image.

**Important:** `fer2013.csv` is not bundled in this repository. Obtain the dataset from a source you are authorized to use, then select it in the application. A saved model may also be loaded as a `.pkl` or `.joblib` file. Only load model files you trust.

### 3. Earthquake-Zone Analysis — Clustering

- Uses the bundled earthquake catalogue or a user-provided CSV.
- Clusters events by geographic location and depth.
- Supports **K-Means** and **Agglomerative (Ward)** clustering.
- Visualizes cluster locations and profiles.
- Helps explore the effect of the selected number of clusters.
- Reports clustering-quality metrics, including Silhouette Score, Davies–Bouldin Index, and Calinski–Harabasz Index.
- Provides options to locate a new earthquake and export results where available.

For a user-provided CSV, the application expects `Latitude`, `Longitude`, and `Depth` columns. `Magnitude` and `Type` may also be used if present. Column matching is case-insensitive for the required geographic fields.

## Datasets

| Dataset | Location | Purpose |
|---|---|---|
| Combined crop-yield data | `data/crop_yield/yield_df.csv` | Main dataset for the regression GUI |
| Crop-yield source tables | `data/crop_yield/` | Rainfall, temperature, pesticide, and yield data used in the notebook |
| Earthquake catalogue | `data/earthquake/database.csv` | Default dataset for clustering |
| FER2013 | User-supplied; not included | Training facial-emotion classifiers |

Keep the supplied directory structure intact so the application can locate the bundled CSV files.

## Requirements

- Python 3.10 or later is recommended.
- `pip`
- The Python packages listed in `requirements.txt`

The project uses Python libraries including Streamlit, NumPy, pandas, Matplotlib, Seaborn, scikit-learn, SciPy, Pillow, Joblib, and Jupyter.

## Setup and Run

### 1. Extract the project

Extract `ML-Capstone-Project-main.zip`, then open a terminal in the extracted `ML-Capstone-Project-main` directory.

### 2. Create and activate a virtual environment (recommended)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

Streamlit should display a local URL in the terminal, usually `http://localhost:8501`. Open that URL in your browser.

To stop the application, press **Ctrl+C** in the terminal.

### 5. Run the notebooks (optional)

Start Jupyter from the project directory:

```bash
jupyter notebook
```

Open a notebook from `notebooks/` and run its cells in order. If a notebook uses a dataset path or file that is not present on your machine, update that path to the correct local location before running it.

## How to Use the Application

1. Run `streamlit run app.py`.
2. Select one of the three tabs: **Regression**, **Classification**, or **Clustering**.
3. Use the bundled dataset when available, or upload a CSV in the relevant section.
4. Choose the model or clustering settings offered by the interface.
5. Train the model or run the analysis.
6. Review the metrics, plots, predictions, and tables.
7. Use the available download controls to save results.

### CSV input reminders

- **Crop yield:** include the target and feature columns listed in the regression section above.
- **Emotion recognition:** use FER2013-style `emotion` and `pixels` columns.
- **Earthquakes:** include `Latitude`, `Longitude`, and `Depth`.

## Machine-Learning Methods and Evaluation

The application offers multiple model choices rather than relying on a single algorithm.

### Regression

Models available in the GUI include Random Forest, K-Nearest Neighbors (KNN), Decision Tree, Gradient Boosting, Ridge Regression, and Linear Regression.

- **R²:** measures how much target variation is explained by the model; higher is generally better.
- **RMSE:** penalizes larger prediction errors more strongly; lower is better.
- **MAE:** the average absolute prediction error; lower is better.

### Classification

The GUI includes several classifier options, with preprocessing for the FER2013 pixel data. Classification performance can be assessed using measures such as accuracy, F1-score, a confusion matrix, and a classification report.

### Clustering

K-Means and Agglomerative (Ward) clustering are available.

- **Silhouette Score:** higher values generally indicate better separation.
- **Davies–Bouldin Index:** lower values generally indicate better-separated clusters.
- **Calinski–Harabasz Index:** higher values generally indicate denser, better-separated clusters.

Metric values depend on the selected data, preprocessing, algorithm, and parameters. They should be interpreted alongside the plots and the characteristics of each dataset.

## Notebooks

| Notebook | Focus |
|---|---|
| `01_Classification_Emotion.ipynb` | Facial-emotion recognition using FER2013 |
| `02_Regression_Crop_Yield.ipynb` | Crop-yield regression and model comparison |
| `03_Clustering_Earthquakes_Corrected.ipynb` | Earthquake clustering and evaluation |

The notebooks document the analysis and experimentation behind the project. The Streamlit GUI is launched separately through `app.py`.

## Troubleshooting

- **`streamlit: command not found`** — activate your virtual environment and run `python -m pip install -r requirements.txt`. You can also launch with `python -m streamlit run app.py`.
- **Bundled CSV not found** — confirm you are running the command from the extracted project directory and that the `data/` folders have not been moved.
- **FER2013 file not found** — this dataset is not included; upload your own `fer2013.csv` in the Classification tab.
- **Notebook cannot find a file** — notebook paths may be relative to the notebook's working directory. Check the path and adjust it for your environment.
- **Training is slow or runs out of memory** — reduce the number of FER2013 images used for training, if that control is available, and close other memory-intensive applications.

## Limitations

- Results depend on dataset quality, preprocessing, feature selection, and model parameters.
- FER2013 is not included, so emotion-model training requires a separate dataset download or a trusted saved model.
- A model's predictions are estimates, not guarantees. In particular, facial-expression classification should not be treated as a reliable determination of a person's actual emotional state.
- The project is intended for academic learning and demonstration, not as a validated production decision system.

## Team

**Group:** DATA MINDS

## License and Data Attribution

No license file is included in the project archive. Add a license if you intend to distribute or reuse the code. Check and follow the license and attribution requirements for each dataset you use, especially any externally obtained FER2013 data.
