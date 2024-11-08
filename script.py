import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import plotly.io as pio
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score, roc_curve, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier 
from sklearn.datasets import load_iris, make_regression

if not os.path.exists('out'):
    os.makedirs('out')

file_path = 'out/score.txt'

def write_metrics_to_file(metrics_dict, file_path='out/score.txt'):
    """
    Écrit les métriques dans un fichier texte.

    Args:
        metrics_dict (dict): Dictionnaire contenant les métriques à écrire.
        file_path (str): Chemin du fichier où les métriques seront écrites.
    """
    with open(file_path, 'a') as file:
        for metric_name, value in metrics_dict.items():
            file.write(f"{metric_name}: {value}\n")
    print(f"Métriques sauvegardées dans {file_path}")

# Définir l'URI de suivi - par défaut, ou local pour éviter des problèmes de chemin
tracking_uri = mlflow.get_tracking_uri()
print(f"Tracking URI: {tracking_uri}")

# Utiliser un tracking URI par défaut pour éviter des erreurs
mlflow.set_tracking_uri(tracking_uri)

# Exemple de modèle
model = LinearRegression()

# Démarrer une nouvelle session de suivi
with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    

# Activer l'autologging MLflow pour toutes les librairies prises en charge
mlflow.autolog()

# Fonction pour charger les données
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Données chargées. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Erreur : fichier {file_path} introuvable.")
        sys.exit(1)
 
# Prétraitement des données
def preprocess_data(df):
    features = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'avg_temp']
    target = 'pesticides_tonnes'
    # Normalisation des features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features, target

# Programme principal
def main(file_path):
    # Charger et préparer les données
    df = load_data(file_path)
    df, features, target = preprocess_data(df)
    
    # Séparer les features et la target
    X = df[features]
    y = df[target]
    
    # Séparer en ensemble d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Démarrer le run MLflow avec un nom personnalisé
    with mlflow.start_run(run_name="Régression Linéaire Baseline"):
        # Entraîner le modèle de régression linéaire
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Faire des prédictions et calculer les métriques
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Enregistrer manuellement les métriques dans MLflow
        mlflow.log_metric("R2 Score", r2)
        mlflow.log_metric("RMSE", rmse)
        
        # Afficher les métriques
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        metrics = {
        'Model': 'Linear Regression Baseline',
        'R2 Score': r2,
        'RMSE': rmse
        }
        write_metrics_to_file(metrics)

if __name__ == "__main__":
    # Permet de passer le chemin du fichier en argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/yield_df.csv"  # Valeur par défaut ou demande d'un chemin personnalisé

    main(file_path)

# Définir le nom de l'expérience MLflow
mlflow.set_experiment("Regression Linéaire avec Cross-Validation")

# Activer l'autologging MLflow pour scikit-learn
mlflow.autolog()

# Fonction pour charger les données
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Données chargées. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        sys.exit(1)

# Prétraitement des données
def preprocess_data(df):
    features = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'avg_temp']
    target = 'pesticides_tonnes'
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features, target

# Entraînement et évaluation du modèle avec MLflow
def train_and_evaluate_model(X, y):
    # Séparation en ensembles d'entraînement + validation et de test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Démarrer une nouvelle exécution MLflow
    with mlflow.start_run(run_name="Régression Linéaire avec Cross-Validation"):
        # Cross-validation (KFold)
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        model = LinearRegression()

        # Obtenir les scores de validation croisée
        cv_scores = cross_val_score(model, X_train_val, y_train_val, cv=kfold, scoring='r2')

        # Enregistrer les scores de validation croisée dans MLflow
        for fold_idx, score in enumerate(cv_scores):
            mlflow.log_metric(f"R2 Fold {fold_idx+1}", score)

        # Entraîner le modèle sur l'ensemble d'entraînement complet
        model.fit(X_train_val, y_train_val)

        # Évaluation sur l'ensemble de test
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Enregistrer les métriques finales dans MLflow
        mlflow.log_metric("R2 Score Test", r2)
        mlflow.log_metric("RMSE Test", rmse)
        mlflow.log_metric("R2 CV Mean", np.mean(cv_scores))

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(model, "LinearRegressionModel")

        # Afficher les métriques
        print(f"Scores de validation croisée R2: {cv_scores}")
        print(f"Score R2 moyen de validation croisée: {np.mean(cv_scores):.4f}")
        print(f"Score R2 sur l'ensemble de test: {r2:.4f}")
        print(f"RMSE sur l'ensemble de test: {rmse:.4f}")
        
        metrics = {
        'Model': 'Linear Regression with Cross-Validation',
        'R2 Score Test': r2,
        'RMSE Test': rmse,
        'R2 CV Mean': np.mean(cv_scores)
        }
        write_metrics_to_file(metrics)

# Fonction principale
def main(file_path):
    df = load_data(file_path)
    df, features, target = preprocess_data(df)
    X = df[features]
    y = df[target]
    train_and_evaluate_model(X, y)

if __name__ == "__main__":
    # Permet de passer le chemin du fichier en argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/yield_df.csv"  # Valeur par défaut ou demande d'un chemin personnalisé

    main(file_path)

# Définir le nom de l'expérience MLflow
mlflow.set_experiment("Comparaison de Modèles de Régression")

# Activer l'autologging MLflow pour scikit-learn
mlflow.autolog()

# Chargement des données
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Données chargées. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        sys.exit(1)

# Prétraitement des données
def preprocess_data(df):
    features = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'avg_temp', 'Item', 'Area']
    target = 'pesticides_tonnes'
    # One-Hot Encoding des colonnes catégorielles
    df = pd.get_dummies(df, columns=['Item', 'Area'], drop_first=True)
    features = [col for col in df.columns if col != target]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features, target

# Entraînement et évaluation des modèles
def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Cross-validation (5-fold)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Entraîner le modèle
            model.fit(X_train, y_train)
            
            # Faire des prédictions sur l'ensemble de test
            y_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Enregistrer les résultats dans MLflow
            mlflow.log_metric("Mean CV R2", np.mean(cv_scores))
            mlflow.log_metric("Test R2 Score", test_r2)
            mlflow.log_metric("Test RMSE", test_rmse)
            
            # Enregistrer le modèle dans MLflow
            mlflow.sklearn.log_model(model, f"{name}_Model")

            # Affichage des résultats
            print(f"\n{name}:")
            print(f"Cross-validation R2 scores: {cv_scores}")
            print(f"Mean CV R2 Score: {np.mean(cv_scores):.4f}")
            print(f"Test R2 Score: {test_r2:.4f}")
            print(f"Test RMSE: {test_rmse:.4f}")
            
            results[name] = {
                'cv_scores': cv_scores,
                'mean_cv_r2': np.mean(cv_scores),
                'test_r2': test_r2,
                'test_rmse': test_rmse
            }
            metrics = {
            'Model': name,
            'Mean CV R2': np.mean(cv_scores),
            'Test R2 Score': test_r2,
            'Test RMSE': test_rmse
            }
            write_metrics_to_file(metrics)
    return results

# Fonction principale
def main(file_path):
    df = load_data(file_path)
    df, features, target = preprocess_data(df)
    X = df[features]
    y = df[target]
    train_and_evaluate_models(X, y)

if __name__ == "__main__":
    # Permet de passer le chemin du fichier en argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/yield_df.csv"  # Valeur par défaut ou demande d'un chemin personnalisé

    main(file_path)

# Optionnel : Configurez Plotly pour afficher les graphiques dans le navigateur
pio.renderers.default = "browser"

# Définir le nom de l'expérience MLflow
mlflow.set_experiment("Optimisation Random Forest et Gradient Boosting avec Optuna")

# Activer l'autologging MLflow pour scikit-learn
mlflow.autolog()

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Données chargées. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        sys.exit(1)

def preprocess_data(df, target='pesticides_tonnes'):
    df = pd.get_dummies(df, columns=['Item', 'Area'], drop_first=True, sparse=True)
    features = [col for col in df.columns if col != target]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df, features, target

# Fonction d'optimisation pour Random Forest et Gradient Boosting
def optimize_rf(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 10, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestRegressor(**param)
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
    return score

def optimize_gb(trial, X, y):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'random_state': 42
    }
    model = GradientBoostingRegressor(**param)
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()
    return score

def optimize_model(model_type, X, y, n_trials=20):
    # Créer une étude Optuna
    study = optuna.create_study(direction='maximize')

    # Définir la fonction objectif en incluant le démarrage d'un run MLflow
    def objective(trial):
        # Démarrer un nouveau run MLflow pour chaque essai
        with mlflow.start_run(nested=True):
            if model_type == 'RandomForest':
                # Définir les hyperparamètres à optimiser
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 10, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestRegressor(**param)
            elif model_type == 'GradientBoosting':
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**param)
            else:
                raise ValueError("Type de modèle non supporté.")

            # Effectuer la validation croisée
            score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error').mean()

            # Enregistrer les hyperparamètres et le score dans MLflow
            mlflow.log_params(param)
            mlflow.log_metric('MSE', -score)

            return score

    # Démarrer un run MLflow pour l'optimisation globale
    with mlflow.start_run(run_name=f"Optimisation {model_type}"):
        study.optimize(objective, n_trials=n_trials)

        # Enregistrer les meilleurs hyperparamètres
        mlflow.log_params(study.best_params)
        mlflow.log_metric(f"Best_MSE_{model_type}", -study.best_value)

        print(f"\nMeilleurs paramètres pour {model_type}:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        metrics = {
        f'Best MSE {model_type}': -study.best_value
        }
        write_metrics_to_file(metrics)

    return study.best_params, study


# Entraînement et évaluation du modèle
def train_model(model_class, params, X_train, y_train):
    model = model_class(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse

# Importance des caractéristiques
def plot_feature_importance(model, feature_names, title="Importance des caractéristiques (Top 20)"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.tight_layout()
    plt.show()

# Analyse de l'importance par permutation
def permutation_importance_analysis(model, X_test, y_test, feature_names, title="Importance par permutation (Top 20)"):
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()[-20:]
    plt.figure(figsize=(12, 8))
    plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=[feature_names[i] for i in sorted_idx])
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Fonction principale
def main(file_path):
    df = load_data(file_path)
    df, features, target = preprocess_data(df)
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optimisation des hyperparamètres
    rf_params, rf_study = optimize_model('RandomForest', X_train, y_train, n_trials=20)
    gb_params, gb_study = optimize_model('GradientBoosting', X_train, y_train, n_trials=20)
    
    # Entraînement des modèles optimisés
    rf_model = train_model(RandomForestRegressor, rf_params, X_train, y_train)
    gb_model = train_model(GradientBoostingRegressor, gb_params, X_train, y_train)
    
    # Évaluation et visualisation des résultats
    models = {'Random Forest': rf_model, 'Gradient Boosting': gb_model}
    for name, model in models.items():
        with mlflow.start_run(run_name=f"Évaluation {name}"):
            r2, rmse = evaluate_model(model, X_test, y_test)
            mlflow.log_metric("R2 Score", r2)
            mlflow.log_metric("RMSE", rmse)
            mlflow.sklearn.log_model(model, f"{name}_Model")
            plot_feature_importance(model, features)
            permutation_importance_analysis(model, X_test, y_test, features)

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/yield_df.csv"
    main(file_path)

# Chargement des données
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Données chargées. Shape:", df.shape)
        return df
    except FileNotFoundError:
        print(f"Erreur : le fichier {file_path} est introuvable.")
        sys.exit(1)

# Prétraitement des données
def preprocess_data(df):
    features = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'avg_temp', 'Item', 'Area']
    target = 'pesticides_tonnes'
    df = pd.get_dummies(df, columns=['Item', 'Area'], drop_first=True)
    features = [col for col in df.columns if col != target]
    return df, features, target

# Génération des courbes d'apprentissage
def plot_learning_curve(model, X_train, y_train, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error',
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Train RMSE')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Test RMSE')
    plt.title(f'Courbe d\'apprentissage - {title}')
    plt.xlabel('Nombre d\'échantillons d\'entraînement')
    plt.ylabel('RMSE')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Entraînement et évaluation des modèles
def train_and_evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=295, max_depth=26, min_samples_split=6, min_samples_leaf=1, random_state=42
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=236, max_depth=10, learning_rate=0.1877
        )
    }
    
    results = {}
    
    for name, model in models.items():
        cv_scores_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        cv_scores_rmse = -cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            'cv_scores_r2': cv_scores_r2,
            'mean_cv_r2': np.mean(cv_scores_r2),
            'cv_scores_rmse': cv_scores_rmse,
            'mean_cv_rmse': np.mean(cv_scores_rmse),
            'test_r2': test_r2,
            'test_rmse': test_rmse
        }
        
        print(f"\n{name}:")
        print(f"Cross-validation R2 scores: {cv_scores_r2}")
        print(f"Mean CV R2 Score: {np.mean(cv_scores_r2):.4f}")
        print(f"Mean CV RMSE Score: {np.mean(cv_scores_rmse):.4f}")
        print(f"Test R2 Score: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        # Générer la courbe d'apprentissage
        plot_learning_curve(model, X_train, y_train, name)
    
    return results

# Fonction principale
def main(file_path):
    # Chargement des données
    df = load_data(file_path)
    
    # Prétraitement des données
    df, features, target = preprocess_data(df)
    X = df[features]
    y = df[target]
    
    # Entraîner et évaluer les modèles
    results = train_and_evaluate_models(X, y)

if __name__ == "__main__":
    # Permet de passer le chemin du fichier en argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "data/yield_df.csv"  # Valeur par défaut

    main(file_path)

# Génération d'un exemple de jeu de données
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model_with_cv(model, X, y):
    # Cross-validation avec 10 plis
    cv_scores_r2 = cross_val_score(model, X, y, cv=10, scoring='r2')
    cv_scores_rmse = -cross_val_score(model, X, y, cv=10, scoring='neg_root_mean_squared_error')
    
    print(f"Cross-validation (10 plis) R2 scores: {cv_scores_r2}")
    print(f"Mean CV R2 Score: {np.mean(cv_scores_r2):.4f}")
    print(f"Mean CV RMSE Score: {np.mean(cv_scores_rmse):.4f}")

# Évaluation du modèle avec les hyperparamètres spécifiés
model = RandomForestRegressor(n_estimators=295, max_depth=26, min_samples_split=6, min_samples_leaf=1, random_state=42)
evaluate_model_with_cv(model, X_train, y_train)

# Chargement des données
file_path = "data/yield_df.csv"  # Assurez-vous que ce chemin est correct
df = pd.read_csv(file_path)

# Visualisation de la distribution des quantités de pesticides
plt.figure(figsize=(10, 6))
sns.histplot(df['pesticides_tonnes'], bins=30, kde=True)
plt.axvline(df['pesticides_tonnes'].median(), color='r', linestyle='--', label=f'Médiane: {df["pesticides_tonnes"].median():.2f}')
plt.axvline(df['pesticides_tonnes'].mean(), color='g', linestyle='--', label=f'Moyenne: {df["pesticides_tonnes"].mean():.2f}')
plt.title('Distribution des quantités de pesticides (tonnes)')
plt.xlabel('Quantité de pesticides (tonnes)')
plt.ylabel('Fréquence')
plt.legend()
plt.show()

# Chargement des données
file_path = "data/yield_df.csv"  # Assurez-vous que ce chemin est correct
df = pd.read_csv(file_path)

# Définir la limite comme la moyenne des quantités de pesticides
limit = df['pesticides_tonnes'].mean()

# Créer une colonne de classification
df['pesticides_class'] = (df['pesticides_tonnes'] > limit).astype(int)

# Vérifier la répartition des classes
class_counts = df['pesticides_class'].value_counts()
print("Répartition des classes dans 'pesticides_class' :")
print(class_counts)

# Afficher la proportion des classes
class_proportion = class_counts / len(df) * 100
print("\nProportion des classes :")
print(class_proportion)

df

# Chargement des données
file_path = "data/yield_df.csv"  # Assurez-vous que ce chemin est correct
df = pd.read_csv(file_path)

# Vérifier et encoder les colonnes catégorielles si elles sont présentes
if 'Area' in df.columns and 'Item' in df.columns:
    # Encodage One-Hot des colonnes catégorielles (Area, Item)
    df = pd.get_dummies(df, columns=['Area', 'Item'], drop_first=True)
else:
    print("Les colonnes 'Area' ou 'Item' ne sont pas présentes dans le DataFrame.")

# Sélection des features et de la cible
X = df.drop(['pesticides_tonnes', 'pesticides_class'], axis=1)  # Features
y = df['pesticides_class']  # Target

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Affichage de la forme des ensembles d'entraînement et de test
print("Shape de X_train:", X_train.shape)
print("Shape de X_test:", X_test.shape)
print("Shape de y_train:", y_train.shape)
print("Shape de y_test:", y_test.shape)

# Définir l'expérience pour la classification
mlflow.set_experiment("Classification")

# Initialisation des modèles
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
}

# Activer l'autologging MLflow pour scikit-learn
mlflow.autolog()

# Validation croisée (StratifiedKFold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Stocker les résultats
results = {}

# Parcourir les modèles
for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name}"):
        print(f"Évaluation du modèle : {model_name}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"Validation croisée (Accuracy) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Enregistrer les résultats de la validation croisée dans MLflow
        mlflow.log_metric("Mean CV Accuracy", cv_scores.mean())
        mlflow.log_metric("CV Accuracy STD", cv_scores.std())
        
        # Entraîner le modèle sur l'ensemble d'entraînement complet
        model.fit(X_train, y_train)
        
        # Prédire sur l'ensemble de test
        y_pred = model.predict(X_test)
        
        # Accuracy sur l'ensemble de test
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy sur le test : {accuracy:.4f}")
        mlflow.log_metric("Test Accuracy", accuracy)
        
        # Rapport de classification
        print(classification_report(y_test, y_pred))
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matrice de confusion - {model_name}')
        plt.ylabel('Vraies classes')
        plt.xlabel('Classes prédites')
        
        # Enregistrer la matrice de confusion localement
        confusion_matrix_path = f"{model_name}_confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.show()
        
        # Enregistrer la matrice de confusion dans MLflow
        mlflow.log_artifact(confusion_matrix_path)
        
        # Supprimer le fichier local de matrice de confusion après l'enregistrement
        os.remove(confusion_matrix_path)
        
        # Calcul de l'AUC-ROC
        roc_auc = roc_auc_score(y_test, y_pred)
        print(f"AUC-ROC : {roc_auc:.4f}")
        mlflow.log_metric("AUC-ROC", roc_auc)
        
        # Tracer la courbe ROC
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'Courbe ROC - {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        plt.show()
        
        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(model, f"{model_name}_model")
        
        # Stocker les résultats
        results[model_name] = {
            'cv_scores': cv_scores,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
        metrics = {
        'Model': model_name,
        'Mean CV Accuracy': cv_scores.mean(),
        'CV Accuracy STD': cv_scores.std(),
        'Test Accuracy': accuracy,
        'AUC-ROC': roc_auc
        }
        write_metrics_to_file(metrics)

# Comparaison des résultats des modèles
print("\nRésumé des performances des modèles :")
for model_name, res in results.items():
    print(f"\nModèle : {model_name}")
    print(f"  Validation croisée (Accuracy) : {res['cv_scores'].mean():.4f} ± {res['cv_scores'].std():.4f}")
    print(f"  Accuracy sur le test : {res['accuracy']:.4f}")
    print(f"  AUC-ROC : {res['roc_auc']:.4f}")

# Chargement d'un exemple de dataset - Remplacez par votre propre dataset
data = load_iris()
X = data.data
y = data.target

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Définir l'expérience MLflow
mlflow.set_experiment("classification")

# Activer l'autologging de MLflow pour Optuna et scikit-learn
mlflow.autolog()

# Fonction d'optimisation des hyperparamètres avec Optuna
def objective(trial):
    # Définir l'espace de recherche pour les hyperparamètres
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 6)
    }

    # Démarrer une nouvelle exécution MLflow pour chaque essai Optuna
    with mlflow.start_run():
        # Enregistrer les hyperparamètres dans MLflow
        mlflow.log_params(param)
        
        # Modèle XGBoost
        model = XGBClassifier(**param, use_label_encoder=False, eval_metric='logloss')

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Prédictions sur les données de test
        y_pred = model.predict(X_test)

        # Calculer l'accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Enregistrer l'accuracy dans MLflow
        mlflow.log_metric('accuracy', accuracy)

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(model, 'xgb_model')

        return accuracy

# Créer une étude Optuna et trouver les meilleurs hyperparamètres
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Afficher les meilleurs hyperparamètres via best_trial
print(f"Meilleurs hyperparamètres : {study.best_trial.params}")

# Génère un graphique montrant l'importance relative de chaque hyperparamètre optimisé
plot_param_importances(study)

# Activer l'autologging de MLflow
mlflow.autolog()

# Définir les hyperparamètres mis à jour
params = {
     'n_estimators': 118, 
    'max_depth': 6, 
    'learning_rate': 0.13834302459712805, 
    'subsample': 0.8607285970463375, 
    'colsample_bytree': 0.6941420232543889, 
    'gamma': 0.2964279221305521, 
    'min_child_weight': 3,
    'random_state' : 42,             
    'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1])  
}

# Démarrer une nouvelle exécution MLflow
with mlflow.start_run(run_name="XGBoost Classifier"):
    try:
        # Initialiser et entraîner le modèle XGBoost
        xgb_model = XGBClassifier(**params)
        xgb_model.fit(X_train, y_train)

        # Prédire sur l'ensemble de test
        y_pred = xgb_model.predict(X_test)

        # Évaluation du modèle
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"\nAccuracy : {accuracy:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(classification_report(y_test, y_pred))

        # Enregistrer les métriques dans MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)

        # plot Matrice de confusion
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Matrice de confusion - XGBoost')
        confusion_matrix_path = "confusion_matrix.png"
        plt.savefig(confusion_matrix_path)
        plt.show()

        # Enregistrer la matrice de confusion dans MLflow
        mlflow.log_artifact(confusion_matrix_path)
        os.remove(confusion_matrix_path)  # Supprimer le fichier après l'enregistrement

        # AUC-ROC
        roc_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
        print(f"AUC-ROC : {roc_auc:.4f}")
        mlflow.log_metric("AUC-ROC", roc_auc)

        # Courbe ROC
        fpr, tpr, thresholds = roc_curve(y_test, xgb_model.predict_proba(X_test)[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Courbe ROC - XGBoost')
        plt.legend()
        roc_curve_path = "roc_curve.png"
        plt.savefig(roc_curve_path)
        plt.show()

        # Enregistrer la courbe ROC dans MLflow
        mlflow.log_artifact(roc_curve_path)
        os.remove(roc_curve_path)  # Supprimer le fichier après l'enregistrement

        # Enregistrer le modèle dans MLflow
        mlflow.sklearn.log_model(xgb_model, "xgb_model")

    except Exception as e:
        print(f"Une erreur est survenue : {e}")


# Prédictions sur l'ensemble d'entraînement
y_train_pred = xgb_model.predict(X_train)

# Évaluation sur l'ensemble d'entraînement
accuracy_train = accuracy_score(y_train, y_train_pred)
f1_train = f1_score(y_train, y_train_pred)
roc_auc_train = roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1])

# Affichage des métriques d'entraînement
print(f"\nTrain Accuracy : {accuracy_train:.4f}")
print(f"Train F1 Score : {f1_train:.4f}")
print(f"Train AUC-ROC : {roc_auc_train:.4f}")

# Enregistrement des métriques d'entraînement dans MLflow
mlflow.log_metric("Train Accuracy", accuracy_train)
mlflow.log_metric("Train F1 Score", f1_train)
mlflow.log_metric("Train AUC-ROC", roc_auc_train)

# Évaluation sur l'ensemble de test (déjà implémentée)
y_test_pred = xgb_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_test_pred)
f1_test = f1_score(y_test, y_test_pred)
roc_auc_test = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

# Affichage des métriques de test
print(f"\nTest Accuracy : {accuracy_test:.4f}")
print(f"Test F1 Score : {f1_test:.4f}")
print(f"Test AUC-ROC : {roc_auc_test:.4f}")

# Enregistrement des métriques de test dans MLflow (si pas déjà enregistré)
mlflow.log_metric("Test Accuracy", accuracy_test)
mlflow.log_metric("Test F1 Score", f1_test)
mlflow.log_metric("Test AUC-ROC", roc_auc_test)

# Comparaison des scores entre entraînement et test
print(f"\nDifférence d'accuracy entre Train et Test : {accuracy_train - accuracy_test:.4f}")
print(f"Différence de F1 score entre Train et Test : {f1_train - f1_test:.4f}")
print(f"Différence d'AUC-ROC entre Train et Test : {roc_auc_train - roc_auc_test:.4f}")

metrics = {
    'Train Accuracy': accuracy_train,
    'Train F1 Score': f1_train,
    'Train AUC-ROC': roc_auc_train,
    'Test Accuracy': accuracy_test,
    'Test F1 Score': f1_test,
    'Test AUC-ROC': roc_auc_test
}
write_metrics_to_file(metrics)

def plot_learning_curve(estimator, X, y, title="Courbe d'apprentissage"):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
    )

    # Calcul des scores moyens
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    # Tracé de la courbe d'apprentissage
    plt.figure()
    plt.title(title)
    plt.xlabel("Nombre d'échantillons d'entraînement")
    plt.ylabel("F1 Score")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Train F1 Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Test F1 Score")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

    # Enregistrement de la courbe d'apprentissage dans MLflow
    plot_path = "learning_curve.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close()  # Fermer la figure pour éviter d'afficher la courbe dans les environnements avec répétitions de graphiques
    
    metrics = {
    'Model': 'Learning Curve',
    'Train Sizes': list(train_sizes),
    'Train Scores Mean': list(train_scores_mean),
    'Test Scores Mean': list(test_scores_mean)
    }
    write_metrics_to_file(metrics)

# Affichage de la courbe d'apprentissage
plot_learning_curve(xgb_model, X_train, y_train)

# Validation croisée à 10 plis
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=10, scoring='accuracy')

# Affichage des résultats
print(f"Validation croisée (10 plis) - Accuracy moyenne : {cv_scores.mean():.4f}")
print(f"Validation croisée (10 plis) - Accuracy écart-type : {cv_scores.std():.4f}")

# Enregistrement des résultats dans MLflow
mlflow.log_metric("Mean CV Accuracy", cv_scores.mean())
mlflow.log_metric("CV Accuracy STD", cv_scores.std())
