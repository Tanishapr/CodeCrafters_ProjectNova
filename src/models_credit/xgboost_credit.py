import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

def train_credit_model(username, password, database, return_details=False):
    """
    Simplified version: trains model, evaluates, saves model & predictions.
    Removed the "bias before/after" sections.

    Returns:
      - if return_details=False: df
      - if return_details=True: (df, (rmse, r2), fi, bias_before, bias_after)
    """
    host = "localhost"
    port = 5432
    connection_str = f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}'
    engine = create_engine(connection_str)

    # --------------------------
    # 1. Load data
    # --------------------------
    df = pd.read_sql('SELECT * FROM "Nova_Partner_Clustered"', engine)
    if df.empty:
        raise ValueError("Clustered data table Nova_Partner_Clustered is empty or missing!")

    # --------------------------
    # 1.a Normalize and remove case-insensitive duplicate columns
    # --------------------------
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    seen = set()
    keep = []
    for c in df.columns:
        key = c.lower() if isinstance(c, str) else c
        if key not in seen:
            seen.add(key)
            keep.append(c)
        else:
            print(f"Warning: dropping duplicate (case-insensitive) column: {c}")
    df = df[keep]

    def find_col(df, *cands):
        lower_map = {c.lower().strip(): c for c in df.columns if isinstance(c, str)}
        for cand in cands:
            if cand is None:
                continue
            k = cand.lower().strip()
            if k in lower_map:
                return lower_map[k]
        return None

    # --------------------------
    # 2. Create single 'gender' column only when truly absent
    # --------------------------
    if find_col(df, "gender") is None:
        gender_ohe_cols = [c for c in df.columns if isinstance(c, str) and c.lower().startswith("gender_")]
        if gender_ohe_cols:
            sub = df[gender_ohe_cols].fillna(0)
            inferred = sub.idxmax(axis=1).str[len("gender_"):].str.replace("_", " ").str.title()
            zero_mask = (sub.sum(axis=1) == 0)
            inferred.loc[zero_mask] = "Other"
            df["gender"] = inferred.astype(str)
            df = df.loc[:, ~df.columns.duplicated()]

    # --------------------------
    # 3. Build synthetic credit_score
    # --------------------------
    np.random.seed(42)
    if find_col(df, "cluster") is None:
        raise ValueError("Required column 'cluster' missing from Nova_Partner_Clustered table.")

    cluster_col = find_col(df, "cluster")
    base_score = df[cluster_col].map({0: 400, 1: 650, 2: 850})

    age_col = find_col(df, "age")
    age_effect = df[age_col] * 1.2 if age_col is not None else 0

    income_col = find_col(df, "earnings_weekly", "income", "earnings")
    income_effect = df[income_col] * 0.02 if income_col is not None else 0

    zone_col = find_col(df, "zone")
    zone_effect = 0
    if zone_col is not None:
        zmap = {"urban": 20, "semi-urban": 0, "semi urban": 0, "rural": -20}
        zone_effect = df[zone_col].astype(str).str.lower().map(zmap).fillna(0)

    df["credit_score"] = (base_score + age_effect + income_effect + zone_effect + np.random.normal(0, 15, len(df))).clip(300, 1000)

    # --------------------------
    # 4. Prepare X, y and numeric feature list
    # --------------------------
    drop_initial = [find_col(df, "credit_score") or "credit_score",
                    find_col(df, "cluster") or "cluster",
                    find_col(df, "partner_id") or "partner_id"]
    X = df.drop(columns=[c for c in drop_initial if c in df.columns], errors="ignore")
    y = df["credit_score"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric features found for model training.")

    preprocessor = ColumnTransformer(transformers=[("num", StandardScaler(), numeric_cols)], remainder="drop")

    # --------------------------
    # 5. Model & pipeline
    # --------------------------
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # --------------------------
    # 6. Evaluate
    # --------------------------
    y_pred = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    print(f"✅ Model Performance: RMSE={rmse:.2f}, R²={r2:.2f}")

    # Feature importance
    try:
        xgb_model = pipeline.named_steps["model"]
        importances = xgb_model.feature_importances_
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    except Exception:
        fi = pd.DataFrame()

    # --------------------------
    # 7. Save model & predictions
    # --------------------------
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "credit_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    df.to_sql('Nova_Partner_CreditModel', engine, if_exists='replace', index=False)
    print("Saved predictions to Nova_Partner_CreditModel")

    # --------------------------
    # 8. Return results (always 5 items for backward compatibility)
    # --------------------------
    if return_details:
        bias_before, bias_after = {}, {}  # placeholders
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