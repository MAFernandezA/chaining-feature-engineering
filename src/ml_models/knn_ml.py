from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler 

class KNN:
    """
    Clase de medición (Scorer) que utiliza K-Nearest Neighbors (KNN) para clasificación.
    Utiliza un pipeline con estandarización, ya que es sensible a la escala.
    """
    def __init__(self, n_neighbors=5):
        # Usamos make_pipeline para aplicar StandardScaler automáticamente
        self.model = make_pipeline(
            StandardScaler(), # Estandariza los datos antes de aplicar KNN
            KNeighborsClassifier(n_neighbors=n_neighbors)
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Entrena el modelo y devuelve la métrica de accuracy."""

        print(f"\n[Scorer: KNN] Entrenando con {self.model.steps[-1][1].n_neighbors} vecinos y escalado estándar...")
        
        # 1. Entrenar (El pipeline escala y luego entrena el KNN)
        self.model.fit(X_train, y_train)
        
        # 2. Predecir
        y_pred = self.model.predict(X_test)
        
        # 3. Evaluar
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"[Scorer: KNN] Entrenamiento finalizado. Accuracy: {accuracy:.4f}")
        
        return accuracy