import numpy as np 
import pandas as pd 
from autofeat import AutoFeatClassifier 
from sklearn.base import BaseEstimator, TransformerMixin



class AutoFeatTransformer(BaseEstimator, TransformerMixin):
    """
    Clase Wrapper para AutoFeatClassifier/Regressor.
    Implementa fit_transform y transform para integrarse al proceso de encadenamiento.
    """
    def __init__(self, n_jobs=-1, verbose=0, **kwargs):
        # Inicializa AutoFeat con parámetros. Usamos n_jobs=-1 para usar todos los cores.
        self.autofeat_model = AutoFeatClassifier(
            n_jobs=n_jobs, 
            verbose=verbose, 
            **kwargs
        )
        self.is_fitted = False

    def fit_transform(self, X, y=None, **fit_params):
        """
        Ajusta AutoFeat (selecciona y genera features) y transforma el conjunto X_train.
        """
        # AutoFeat trabaja mejor con DataFrames o NumPy arrays
        X_df = self._to_dataframe(X)
        
        print(f"\nGenerando features con AutoFeat... (Inicial: {X_df.shape})")
        
        # El método fit_transform de AutoFeat devuelve las nuevas features
        X_new = self.autofeat_model.fit_transform(X_df, y)
        
        self.is_fitted = True
        print(f"Features generadas. Nuevo tamaño: {X_new.shape}")
        
        return X_new

    def transform(self, X):
        """
        Aplica la transformación aprendida (las nuevas features) a un nuevo conjunto (X_test).
        """
        if not self.is_fitted:
            raise RuntimeError("El transformador debe ser ajustado primero con fit_transform.")
            
        X_df = self._to_dataframe(X)
        
        # El método transform solo aplica las features previamente seleccionadas
        X_new = self.autofeat_model.transform(X_df)
        
        return X_new
        
    def _to_dataframe(self, X):
        """Convierte la entrada a DataFrame si es un array de NumPy."""
        if isinstance(X, np.ndarray):
            # Crea nombres de columnas genéricos si no los tiene
            return pd.DataFrame(X, columns=[f'c{i}' for i in range(X.shape[1])])
        return X
