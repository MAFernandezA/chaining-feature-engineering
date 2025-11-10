from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SVM:
    """
    Clase de medición (Scorer) que utiliza Support Vector Machine (SVC) para clasificacion.
    Se utiliza con un pipeline que incluye estandarización.
    """
    def __init__(self, random_state=42, C=1.0, gamma='scale'):
        # Usamos make_pipeline para aplicar StandardScaler automáticamente
        self.model = make_pipeline(
            StandardScaler(),
            SVC(C=C, gamma=gamma, random_state=random_state)
        )

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Entrena el modelo y devuelve la métrica de accuracy."""

        print("\n[Scorer: SVM] Entrenando con escalado estándar...")
        
        # 1. Entrenar (El pipeline escala internamente antes de entrenar el SVC)
        self.model.fit(X_train, y_train)
        
        # 2. Predecir (El pipeline escala internamente antes de predecir)
        y_pred = self.model.predict(X_test)
        
        # 3. Evaluar
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"[Scorer: SVM] Entrenamiento finalizado. Accuracy: {accuracy:.4f}")
        
        return accuracy