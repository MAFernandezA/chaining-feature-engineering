from sklearn.preprocessing import PolynomialFeatures, StandardScaler # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin # type: ignore
import numpy as np # type: ignore

class PolynomialFeatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        # El grado del polinomio es un hiperparámetro
        self.poly_model = PolynomialFeatures(degree=degree, include_bias=False)
        self.scaler_model = StandardScaler()
        self.degree = degree

    def fit_transform(self, X, y=None):
        print(f"Generando features polinomiales (grado={self.degree})...")
        # Aseguramos que la entrada sea NumPy array para consistencia
        X_arr = X if isinstance(X, np.ndarray) else X.values 
        X_scaled = self.scaler_model.fit_transform(X_arr)
        X_poly = self.poly_model.fit_transform(X_scaled)

        return X_poly
    
    def transform(self, X):
        X_arr = X if isinstance(X, np.ndarray) else X.values
        
        # 1. Aplicar ESCALAMIENTO (usando los parámetros aprendidos en fit)
        X_scaled = self.scaler_model.transform(X_arr)
        
        # 2. Aplicar la transformación Polinomial
        X_poly = self.poly_model.transform(X_scaled)
        
        return X_poly
    