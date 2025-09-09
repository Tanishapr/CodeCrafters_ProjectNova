# cc-
## Project Overview
This project implements a comprehensive machine learning pipeline designed for partner segmentation and Nova score prediction, leveraging synthetically generated mock data to simulate real-world operational scenarios. The pipeline orchestrates multiple stages—including data ingestion, preprocessing with feature engineering, clustering via K-Means, and interactive score prediction through a user-friendly Streamlit application. In addition to core model functionalities, the project integrates fairness assessments through statistical bias metrics and advanced explainability techniques using SHAP (SHapley Additive exPlanations), ensuring transparency and trustworthiness of predictions. The modular architecture supports scalability, easy maintainability, and seamless adaptation to real datasets and production environments, making it a robust prototype for delivery partner analytics and decision support.


## Table of Contents
- [Project Overview](#project-overview)
- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)


## Setup & Installation
```git clone <repo-url>
  cd <repo-folder>
  pip install -r requirements.txt
```
Configure your PostgreSQL credentials in the Fairness.ipynb


## Project Structure
```bash
Project/
├── Data/
│   └── Data.csv                    # Mock data CSV file
├── src/
│   ├── models/                     # Stored pickle models
│   │   ├── credit_model_fair.pkl
│   │   ├── credit_model_tuned.pkl
│   │   └── credit_model.pkl
│   ├── models_credit/              # XGBoost credit model script
│   │   └── xgboost_credit.py
│   ├── notebooks/                  # Jupyter notebooks
│   │   └── fairness.ipynb
│   ├── pipeline/
│   │   ├── data_create.py          # Mock data generation script
│   │   ├── data_injest.py          # Data ingestion into PostgreSQL
│   │   ├── data_cluster.py         # Clustering pipeline script
│   │   ├── shap_credit.py          # SHAP explainability analysis
│   │   └── model_predictions.py    # Model prediction utilities
│   └── app.py                      # Streamlit app for user interaction and scoring
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Usage

1. Open a terminal and navigate to the source folder:
2. `pip install -r requirements.txt`
3. To create Mock_Data for this case, run data_create.py
    ```cd src
      cd pipepline
      python data_create.py
   ```
4. To use Streamlit App
   ```cd src
      streamlit run app.py
   ```
5. To view and analyse the existing bias and variance present in dataset, run fairness.ipynb under notebooks

### Using the Streamlit interface:
1. Enter mock database credentials and feature values as prompted.
2. The app will automatically ingest and prepare the data, then display the predicted Nova score.
3. The app also generates and displays SHAP analysis to show feature impacts on the prediction results.
### Fairness and Bias Analysis:
1. To examine fairness metrics and group bias, open and run the fairness.ipynb notebook in Jupyter.
2. This notebook loads the clustered data and outputs bias metrics and distribution charts
   

## Results
1. Partner segmentation and Nova score predictions are provided interactively.
2. Bias metrics and fairness charts are shown in the notebook.
3. SHAP analysis reveals which features most influence model predictions, increasing model transparency and trust.

## Contributors
- [![Harshita Jangde](https://github.com/HarshitaJangde.png?size=40)](https://github.com/HarshitaJangde) [@HarshitaJangde](https://github.com/HarshitaJangde) 
- [![Tanisha Priya](https://github.com/tanishapr.png?size=40)](https://github.com/tanishapr) [@TanishaPriya](https://github.com/tanishapr) 
- [![Prachi Tavse](https://github.com/prachitavse.png?size=40)](https://github.com/prachitavse) [@PrachiTavse](https://github.com/prachitavse) 