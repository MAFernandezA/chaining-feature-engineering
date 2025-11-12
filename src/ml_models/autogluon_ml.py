import pandas as pd 
from autogluon.tabular import TabularPredictor 
import numpy as np 

class AutoGluon:
    """
    Clase de medición (Scorer) que utiliza AutoGluon para entrenar y evaluar
    la calidad del dataset con las features actuales.
    """
    def __init__(self, time_limit=30, random_state=42, ask_for_gpu=True):
        # Time_limit es crucial para controlar el tiempo que tarda AutoGluon
        self.time_limit = time_limit
        self.label_column = "fe_label" # Nombre temporal para la columna de etiquetas
        self.predictor = None
        self.ag_args_fit_gpu = {}

        if ask_for_gpu:
            use_gpu = input("¿Desea utilizar la GPU (tarjeta gráfica) para acelerar el entrenamiento? (s/n): ").lower().strip()
            
            if use_gpu == 's':
                print("GPU habilitada para AutoGluon.")
                self.ag_args_fit_gpu = {
                    'num_gpus': 1 
                }
            else:
                print("GPU deshabilitada para AutoGluon. Usando solo CPU.")
                self.ag_args_fit_gpu = {
                    'num_gpus': 0
                }
        else:
            # Opción por defecto (sin preguntar, se puede configurar desde main.py)
            self.ag_args_fit_gpu = {'num_gpus': 0}

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """
        Entrena AutoGluon con los conjuntos de entrenamiento y evalúa con el conjunto de prueba.
        """
        print(f"\n[Scorer: AutoGluon (GLOBAL)] Entrenando con límite de tiempo: {self.time_limit}s...")
        
        # 1. Preparar DataFrames para AutoGluon
        # AutoGluon requiere un DataFrame que contenga las features Y las etiquetas
        
        # X_train puede ser np.ndarray o DataFrame, lo convertimos siempre a DataFrame
        if isinstance(X_train, np.ndarray):
             X_train_df = pd.DataFrame(X_train)
        else:
             X_train_df = pd.DataFrame(X_train.copy())

        # El target (y_train) debe ser añadido al DataFrame de entrenamiento
        train_data = X_train_df.copy()
        train_data[self.label_column] = y_train


        # 2. Inicializar y Entrenar AutoGluon
        self.predictor = TabularPredictor(
            label=self.label_column,
            eval_metric="accuracy",
            path=None, 
            verbosity=0
        ).fit(
            train_data,
            presets="best_quality",
            time_limit=self.time_limit,
            ag_args_fit=self.ag_args_fit_gpu
        )

        # 3. Preparar DataFrame de Prueba
        if isinstance(X_test, np.ndarray):
             X_test_df = pd.DataFrame(X_test)
        else:
             X_test_df = pd.DataFrame(X_test.copy())
             
        test_data = X_test_df.copy()
        test_data[self.label_column] = y_test
        leaderboard = self.predictor.leaderboard(test_data, silent=True)

        base_models_lb = leaderboard[~leaderboard['model'].str.contains('Ensemble', case=False, na=False)]

        if base_models_lb.empty:
            print("[AutoGluon] Advertencia: No se encontraron modelos base. Usando el mejor score global.")
            average_accuracy = leaderboard['score_val'].max()
        else:
            # Calcular el promedio de la métrica ('score_val' es el Accuracy en este caso)
            average_accuracy = base_models_lb['score_val'].mean()

        print("--- Resumen AutoGluon (Global) ---")
        print(f"Número de modelos base entrenados: {len(base_models_lb)}")
        print(f"Mejor Accuracy de un modelo individual: {leaderboard['score_val'].max():.4f}")
        print(f"Accuracy PROMEDIO (Global Score): {average_accuracy:.4f}")
        print("-----------------------------------")
        
        return average_accuracy
