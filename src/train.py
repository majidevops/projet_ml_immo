import mlflow
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Chargement et préparation des données
data = fetch_california_housing(as_frame=True)
X = data.frame.drop('MedHouseVal', axis=1)
y = data.frame['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Configuration de l'expérience MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Prediction_Immo_Californie")

# 3. Entraînement et suivi des métriques
with mlflow.start_run():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))

    # On enregistre uniquement la métrique
    mlflow.log_metric("MAE", mae)

    # Sauvegarde locale du modèle à la racine
    joblib.dump(model, "model.pkl")

    print(f"Entraînement réussi ! MAE obtenue : {mae:.4f}")
