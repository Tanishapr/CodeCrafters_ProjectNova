import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib  # for saving model

def train_credit_model(username, password, database, return_details=False):
    host = "localhost"
    port = 5432
    connection_str = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)

    # --------------------------
    # 1. Load clustered data from Postgres
    # --------------------------
    df = pd.read_sql('SELECT * FROM "Nova_Partner_Clustered"', engine)
    if df.empty:
        raise ValueError("Clustered data table Nova_Partner_Clustered is empty or missing!")

    # Map clusters -> credit scores (like bureaus)
    cluster_to_score = {
        0: (300, 500),   # low
        1: (500, 750),   # medium
        2: (750, 1000)   # high
    }

    # Assign random scores (simulate real bureau assignment)
    np.random.seed(42)
    df["credit_score"] = df["cluster"].apply(
        lambda c: np.random.randint(cluster_to_score[c][0], cluster_to_score[c][1])
    )

    # --------------------------
    # 2. Features & Labels
    # --------------------------
    X = df.drop(columns=["credit_score", "cluster", "partner_id"], errors="ignore")
    y = df["credit_score"]

    numeric_cols = X.columns.tolist()  # all are numeric now
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols)
        ]
    )

    # --------------------------
    # 3. Train XGBoost Regressor
    # --------------------------
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)

    # --------------------------
    # 4. Evaluation
    # --------------------------
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"✅ Model Performance: RMSE={rmse:.2f}, R²={r2:.2f}")

    # --------------------------
    # 5. Feature Importance
    # --------------------------
    xgb_model = pipeline.named_steps["model"]
    importances = xgb_model.feature_importances_
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

    fi = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    # --------------------------
    # 6. Bias Analysis
    # --------------------------
    df["predicted_score"] = pipeline.predict(X)

    bias_before = None
    if "gender" in df.columns or "zone" in df.columns:
        results = {}
        if "gender" in df.columns:
            results["gender"] = df.groupby("gender")["predicted_score"].mean()
        if "zone" in df.columns:
            results["zone"] = df.groupby("zone")["predicted_score"].mean()
        bias_before = results

    # --------------------------
    # 7. Bias Mitigation (drop sensitive cols)
    # --------------------------
    X_fair = df.drop(columns=["credit_score", "cluster", "partner_id", "gender", "zone"], errors="ignore")
    numeric_cols_fair = X_fair.columns.tolist()

    preprocessor_fair = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols_fair)
        ]
    )

    pipeline_fair = Pipeline(steps=[
        ("preprocessor", preprocessor_fair),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X_fair, y, test_size=0.2, random_state=42
    )
    pipeline_fair.fit(X_train, y_train)
    df["predicted_score_fair"] = pipeline_fair.predict(X_fair)

    bias_after = None
    if "gender" in df.columns or "zone" in df.columns:
        results = {}
        if "gender" in df.columns:
            results["gender"] = df.groupby("gender")["predicted_score_fair"].mean()
        if "zone" in df.columns:
            results["zone"] = df.groupby("zone")["predicted_score_fair"].mean()
        bias_after = results

    # --------------------------
    # 8. Save Model
    # --------------------------
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "credit_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    # --------------------------
    # 9. Save results to Postgres
    # --------------------------
    df.to_sql('Nova_Partner_CreditModel', engine, if_exists='replace', index=False)

    # Return results for Streamlit
    if return_details:
        return df, (rmse, r2), fi, bias_before, bias_after
    return df

# import pandas as pd  
# import numpy as np  
# import matplotlib.pyplot as plt  
# import seaborn as sns  
# from sqlalchemy import create_engine  
# from sklearn.model_selection import train_test_split  
# from sklearn.preprocessing import StandardScaler  
# from sklearn.compose import ColumnTransformer  
# from sklearn.pipeline import Pipeline  
# from sklearn.metrics import mean_squared_error, r2_score  
# import xgboost as xgb 
# import joblib 

# def train_credit_model(username, password, database):  
#     host = "localhost"  
#     port = 5432  
#     connection_str = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'  
#     engine = create_engine(connection_str)  

#     # --------------------------  
#     # 1. Load clustered data from Postgres  
#     # --------------------------  
#     df = pd.read_sql('SELECT * FROM "Nova_Partner_Clustered"', engine)  
#     if df.empty:  
#         raise ValueError("Clustered data table Nova_Partner_Clustered is empty or missing!")  

#     # Map clusters -> credit scores (like bureaus)  
#     cluster_to_score = {  
#         0: (300, 500),   # low  
#         1: (500, 750),   # medium  
#         2: (750, 1000)   # high  
#     }  

#     # Assign a random score in the band  
#     np.random.seed(42)  
#     df["credit_score"] = df["cluster"].apply(
#         lambda c: np.random.randint(cluster_to_score[c][0], cluster_to_score[c][1])  
#     )  

#     print("✅ Step 1: Data loaded & credit_score assigned")  

#     # --------------------------  
#     # 2. Train/Test split  
#     # --------------------------  
#     X = df.drop(columns=["credit_score", "cluster", "partner_id"], errors="ignore")  
#     y = df["credit_score"]  

#     # all columns are already numeric  
#     numeric_cols = list(X.columns)  

#     preprocessor = ColumnTransformer(  
#         transformers=[  
#             ("num", StandardScaler(), numeric_cols)  
#         ]  
#     )  
#     print("✅ Step 2: Preprocessor defined")  

#     # --------------------------  
#     # 3. Train XGBoost Regressor  
#     # --------------------------  
#     model = xgb.XGBRegressor(  
#         n_estimators=200,  
#         learning_rate=0.1,  
#         max_depth=6,  
#         random_state=42  
#     )  

#     pipeline = Pipeline(steps=[  
#         ("preprocessor", preprocessor),  
#         ("model", model)  
#     ])  

#     X_train, X_test, y_train, y_test = train_test_split(  
#         X, y, test_size=0.2, random_state=42  
#     )  
#     pipeline.fit(X_train, y_train)  
#     print("✅ Step 3: Model trained")  

#     # --------------------------  
#     # 4. Evaluation  
#     # --------------------------  
#     y_pred = pipeline.predict(X_test)  
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))  
#     r2 = r2_score(y_test, y_pred)  
#     print(f"✅ Step 4: Model Performance: RMSE={rmse:.2f}, R²={r2:.2f}")  

#     # --------------------------  
#     # 5. Feature Importance  
#     # --------------------------  
#     xgb_model = pipeline.named_steps["model"]  
#     importances = xgb_model.feature_importances_  
#     feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()  

#     fi = pd.DataFrame(  
#         {"feature": feature_names, "importance": importances}  
#     ).sort_values(by="importance", ascending=False)  

#     plt.figure(figsize=(10,5))  
#     sns.barplot(data=fi.head(10), x="importance", y="feature")  
#     plt.title("Top 10 Feature Importances (XGBoost)")  
#     plt.show()  

#     # --------------------------  
#     # 6. Bias Analysis (reconstruct groups from dummies)  
#     # --------------------------  
#     df["predicted_score"] = pipeline.predict(X)  

#     # Gender bias (from dummy cols)  
#     if "gender_Female" in df.columns and "gender_Male" in df.columns:  
#         df["gender"] = np.where(df["gender_Female"] == 1, "Female", "Male")  
#         bias_gender = df.groupby("gender")["predicted_score"].mean()  
#         print("\n📊 Bias by Gender:\n", bias_gender)  

#         sns.barplot(x=bias_gender.index, y=bias_gender.values)  
#         plt.title("Average Predicted Score by Gender")  
#         plt.show()  

#     # Zone bias (from dummy cols)  
#     zone_cols = [c for c in df.columns if c.startswith("zone_")]  
#     if zone_cols:  
#         df["zone"] = df[zone_cols].idxmax(axis=1).str.replace("zone_", "")  
#         bias_zone = df.groupby("zone")["predicted_score"].mean()  
#         print("\n📊 Bias by Zone:\n", bias_zone)  

#         sns.barplot(x=bias_zone.index, y=bias_zone.values)  
#         plt.title("Average Predicted Score by Zone")  
#         plt.show()  

#     # --------------------------  
#     # 7. Bias Mitigation (drop sensitive features)  
#     # --------------------------  
#     sensitive_cols = [c for c in df.columns if c.startswith("gender_") or c.startswith("zone_")]  
#     X_fair = df.drop(columns=["credit_score", "cluster", "partner_id", "gender", "zone"] + sensitive_cols, errors="ignore")  

#     numeric_cols_fair = list(X_fair.columns)  

#     preprocessor_fair = ColumnTransformer(  
#         transformers=[  
#             ("num", StandardScaler(), numeric_cols_fair)  
#         ]  
#     )  

#     pipeline_fair = Pipeline(steps=[  
#         ("preprocessor", preprocessor_fair),  
#         ("model", model)  
#     ])  

#     X_train, X_test, y_train, y_test = train_test_split(  
#         X_fair, y, test_size=0.2, random_state=42  
#     )  
#     pipeline_fair.fit(X_train, y_train)  

#     df["predicted_score_fair"] = pipeline_fair.predict(X_fair)  

#     if "gender" in df.columns:  
#         bias_gender_fair = df.groupby("gender")["predicted_score_fair"].mean()  
#         print("\n📊 Bias After Mitigation (Gender):\n", bias_gender_fair)  

#     if "zone" in df.columns:  
#         bias_zone_fair = df.groupby("zone")["predicted_score_fair"].mean()  
#         print("\n📊 Bias After Mitigation (Zone):\n", bias_zone_fair)  



#     # --------------------------  
#     # 8. Save results  
#     # --------------------------  
#     df.to_sql('Nova_Partner_CreditModel', engine, if_exists='replace', index=False)  
#     print("✅ Step 8: Results saved to Postgres")  

#     return df