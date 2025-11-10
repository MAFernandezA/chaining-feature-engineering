import numpy as np # type: ignore
import pandas as pd # type: ignore
from boruta import BorutaPy # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore

class BorutaFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Clase Wrapper para BorutaPy que realiza la Selección de Características.
    Se ajusta a la interfaz de Scikit-learn (fit_transform, transform).
    """
    def __init__(self, n_estimators='auto', max_iter=100, **kwargs):
        # El estimador base para Boruta es un RandomForestClassifier
        self.rf_estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=7,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        # Inicializa BorutaPy con el estimador y parámetros
        self.boruta_model = BorutaPy(
            estimator=self.rf_estimator,
            n_estimators='auto',
            max_iter=max_iter, # El límite de iteraciones es un parámetro clave
            random_state=42,
            verbose=0 # Se recomienda silenciar Boruta para el encadenamiento
        )
        self.is_fitted = False
        self.selected_features_mask = None

    def fit_transform(self, X, y=None, **fit_params):
        """
        Ajusta Boruta (selecciona las features) y retorna el dataset reducido.
        """
        X_arr = X if isinstance(X, np.ndarray) else X.values
        
        print(f"\nSeleccionando features con Boruta... (Inicial: {X_arr.shape})")
        
        # 1. Ajustar Boruta para encontrar las mejores features
        self.boruta_model.fit(X_arr, y)
        
        # 2. Guardar la máscara de las features seleccionadas
        self.selected_features_mask = self.boruta_model.support_
        
        # 3. Aplicar la selección al conjunto de entrenamiento
        X_reduced = X_arr[:, self.selected_features_mask]
        
        self.is_fitted = True
        print(f"Features seleccionadas. Nuevo tamaño: {X_reduced.shape}")
        
        return X_reduced

    def transform(self, X):
        """
        Aplica la selección de features aprendida (la misma máscara) al conjunto de prueba.
        """
        if not self.is_fitted:
            raise RuntimeError("El transformador Boruta debe ser ajustado primero con fit_transform.")
            
        X_arr = X if isinstance(X, np.ndarray) else X.values
        
        # 4. Aplicar la selección al conjunto de prueba
        X_reduced = X_arr[:, self.selected_features_mask]
        
        return X_reduced
