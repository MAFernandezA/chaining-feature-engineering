from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForest:
    """
    Modelo base para clasificacion. Usa RandomForest por defecto.
    """
    def __init__(self, random_state=42):
        # Este será el modelo que se entrena en cada iteración
        self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Entrena el modelo y devuelve la métrica de accuracy."""
        
        # 1. Entrenar
        self.model.fit(X_train, y_train)
        
        # 2. Predecir
        y_pred = self.model.predict(X_test)
        
        # 3. Evaluar
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy