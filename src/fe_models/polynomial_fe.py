from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin 
import numpy as np



class PolynomialFeatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        # El grado del polinomio es un hiperparámetro
        self.poly_model = PolynomialFeatures(degree=degree, include_bias=False)
        self.degree = degree

    def fit_transform(self, X, y=None):
        print(f"Generando features polinomiales (grado={self.degree})...")
        # Aseguramos que la entrada sea NumPy array para consistencia
        X_arr = X if isinstance(X, np.ndarray) else X.values
        X_poly = self.poly_model.fit_transform(X_arr)

        return X_poly
    
    def transform(self, X):
        X_arr = X if isinstance(X, np.ndarray) else X.values
        
        # 2. Aplicar la transformación Polinomial
        X_poly = self.poly_model.transform(X_arr)
        
        return X_poly
