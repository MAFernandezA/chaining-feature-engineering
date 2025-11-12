
from src.ml_models.autogluon_ml import AutoGluon
from src.ml_models.randomforest_ml import RandomForest 
from src.ml_models.svm_ml import SVM
from src.ml_models.knn_ml import KNN


def select_ml_scorer():
    
    # Mapeo de opciones a instancias de clases
    available_scorers = {
        1: ("AutoGluon (Alto Rendimiento)", AutoGluon(time_limit=150)),
        2: ("RandomForest (Rápido y Base)", RandomForest()),
        3: ("SVM", SVM()),
        4: ("KNN", KNN())
    }
    
    print("\n==================================================")
    print("SELECCIÓN DEL MODELO SCORER (ML)")
    print("==================================================")
    
    for key, (name, _) in available_scorers.items():
        print(f"[{key}] - {name}")
    
    while True:
        try:
            choice = input("Seleccione el número del modelo ML (Scorer) a utilizar: ")
            index = int(choice)
            if index in available_scorers:
                selected_name, selected_instance = available_scorers[index]
                print(f"Scorer seleccionado: {selected_name}")
                return selected_instance # Retorna la instancia del Scorer
            else:
                print("Opción no válida. Por favor, ingrese un número de la lista.")
        except ValueError:
            print("Entrada no válida. Por favor, ingrese un número entero.")

