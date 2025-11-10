import numpy as np # type: ignore
import pandas as pd# type: ignore
import featuretools as ft # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin # type: ignore

class FeatureToolsTransformer(BaseEstimator, TransformerMixin):
    """
    Clase Wrapper para Featuretools. Genera features sintéticas usando
    operaciones básicas (ADD_NUMERIC, MULTIPLY_NUMERIC).
    """
    def __init__(self, trans_primitives=None, max_depth=1):
        
        # Parámetros por defecto para Featuretools
        self.trans_primitives = trans_primitives or ["add_numeric", "multiply_numeric"]
        self.max_depth = max_depth
        self.feature_defs = None # Almacenará las definiciones de features aprendidas
        
    def fit_transform(self, X, y=None, **fit_params):
        """
        Ajusta Featuretools para aprender las feature_defs y transforma X_train.
        """
        # Featuretools necesita DataFrames con columnas nombradas
        X_df = self._to_dataframe(X, is_train=True)
        
        # 1. Crear EntitySet (ft requiere que los datos estén en este formato)
        es = ft.EntitySet(id="dataset")
        es.add_dataframe(dataframe_name="data", dataframe=X_df, index="index")

        print(f"\nGenerando features con Featuretools... (Inicial: {X_df.shape})")

        # 2. Deep Feature Synthesis (DFS) - Aprender y generar features
        feature_matrix, self.feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="data",
            trans_primitives=self.trans_primitives,
            max_depth=self.max_depth,
            verbose=False # Silenciamos el output para que no inunde la consola
        )
        
        print(f"Features generadas. Nuevo tamaño: {feature_matrix.shape}")
        
        # 3. Retornar las features generadas (excluyendo el índice temporal)
        # El primer elemento de la matriz de features es X_df original (copiado),
        # y los demás son las features sintéticas. Devolvemos solo los valores
        # para que el encadenador no tenga problemas con el tipo de datos.
        return feature_matrix.values

    def transform(self, X):
        """
        Aplica las transformaciones aprendidas (self.feature_defs) a un nuevo conjunto (X_test).
        """
        if self.feature_defs is None:
            raise RuntimeError("El transformador Featuretools debe ser ajustado primero con fit_transform.")
            
        X_df = self._to_dataframe(X, is_train=False)
        
        # 1. Crear EntitySet de prueba
        es = ft.EntitySet(id="test_dataset")
        es.add_dataframe(dataframe_name="data", dataframe=X_df, index="index")

        # 2. Aplicar las feature_defs aprendidas
        feature_matrix = ft.calculate_feature_matrix(
            features=self.feature_defs,
            entityset=es
        )
        
        # 3. Retornar los valores
        return feature_matrix.values
        
    def _to_dataframe(self, X, is_train):
        """Convierte la entrada a DataFrame si es un array de NumPy y añade el índice requerido por ft."""
        if isinstance(X, np.ndarray):
            # Crea nombres de columnas genéricos si no los tiene
            X_df = pd.DataFrame(X, columns=[f'c{i}' for i in range(X.shape[1])])
        else:
            X_df = X
            
        # Featuretools requiere una columna de índice
        X_df['index'] = X_df.index
        
        # Si no es entrenamiento, quitamos la columna 'target' si existe
        if not is_train and 'target' in X_df.columns:
             X_df = X_df.drop(columns=['target'], errors='ignore')
             
        return X_df
