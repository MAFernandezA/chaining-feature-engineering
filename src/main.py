import os
import pandas as pd 
import numpy as np
from datetime import datetime
from src.ml_models.select_scorer import select_ml_scorer
from src.core.reporting import generate_pdf_report
from sklearn.preprocessing import StandardScaler
import time

from src.core.utils import (
    apply_pca_reduction, 
    split_dataset
)

from src.core.fe_chain import (
    FeatureEngineeringChainer
)

from src.core.load_dataframe import (
    select_dataset_file, 
    load_mat_to_dataframe,
    load_csv_to_dataframe,
)



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'estadisticas')

def print_dataset_info(name, X, X_pca=None):
    print("\n--- Información del Dataset ---")
    print(f"Dataset cargado: '{name}'")
    print(f"Tamaño original: {X.shape}")
    
    if X_pca is not None:
        print(f"Tamaño después de PCA: {X_pca.shape}")
    print("-------------------------------\n")

def main():
    print("==================================================")
    print("      INICIO DEL PROCESO DE FEATURE ENGINEERING     ")
    print("==================================================")
    
    # 1. Seleccionar el dataset
    try:
        file_path, file_name = select_dataset_file(DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. Cargar y convertir a DataFrame/Series
    ext = os.path.splitext(file_name)[1].lower()

    if ext == '.mat':
            X, y = load_mat_to_dataframe(file_path)
    elif ext == '.csv':
        X, y = load_csv_to_dataframe(file_path)
    else:
        # Debería ser atrapado por select_dataset_file, pero como seguridad
        print(f"Error: Tipo de archivo '{ext}' no soportado.")
        return
    
    selected_name = file_name
    X_initial_shape = X.shape
    
    # 3. Mostrar información inicial
    print_dataset_info(file_name, X)
    
    # 4. Preguntar por la reducción de dimensionalidad (PCA)
    apply_pca_initial = input(
            "¿Se requiere reducir la dimensionalidad con PCA en el dataset BASE? (s/n): "
        ).lower().strip() == 's'
    
    apply_pca_dynamic = False
    pca_threshold = 99999  # Umbral alto por defecto

    if input("¿Desea aplicar PCA a los datasets generados en el encadenamiento? (s/n): ").lower().strip() == 's':
        apply_pca_dynamic = True
        
        while True:
            try:
                # 3. Umbral de features
                threshold = int(input("Ingrese la cantidad mínima de features para aplicar PCA dinámico: "))
                if threshold < 2:
                    print("El umbral debe ser mayor o igual a 2. Intente de nuevo.")
                    continue
                pca_threshold = threshold
                break
            except ValueError:
                print("Entrada no válida. Por favor, ingrese un número entero.")

    start_time = time.time()

    if apply_pca_initial:
        # Aplicar PCA
        print("... Aplicando Reducción de Dimensionalidad (PCA) en el dataset BASE ...")
        X_processed = apply_pca_reduction(X)
        
        # 5. Mostrar información con PCA
        print_dataset_info(file_name, X, X_processed)
    else:
        print("Omite la reducción de dimensionalidad con PCA.")
        X_processed = X.values # Usamos el valor del DataFrame para consistencia

    print("\n... Aplicando Estandarización (StandardScaler) al Dataset Base ...")
    scaler = StandardScaler()
    X_processed_scaled = scaler.fit_transform(X_processed)
    X_processed = X_processed_scaled
    print(f"Dataset estandarizado. Forma: {X_processed.shape}")

    X_processed_final = X_processed

    # A partir de aquí, X_processed y y están listos para ser usados
    print("Etapa de carga y preprocesamiento de datos finalizada.")
    print(f"Datos listos para usar: X.shape={X_processed.shape}, y.shape={y.shape}")

    X_train, X_test, y_train, y_test = split_dataset(X_processed, y)

    ml_scorer = select_ml_scorer()

    print("\n==================================================")
    print("INICIANDO EL ENCADENAMIENTO DE FEATURES...")
    print("==================================================")

    chainer = FeatureEngineeringChainer(X_train, y_train, X_test, y_test,
                                        apply_pca_dynamic=apply_pca_dynamic,
                                        pca_threshold=pca_threshold,
                                        ml_scorer=ml_scorer)
    
    # Por ahora solo ejecuta la evaluación base y la prueba inicial de Autofeat
    final_results = chainer.run_chaining() 


    end_time = time.time()
    execution_time_seconds = end_time - start_time
    minutes = int(execution_time_seconds // 60)
    seconds = int(execution_time_seconds % 60)
    time_str = f"{minutes}m {seconds}s"

    dataset_info = {
        'name': selected_name,
        'original_shape': X_initial_shape,
        'pca_base_applied': apply_pca_initial,
        'base_dataset_shape': X_processed_final.shape,
        'base_X_train': X_train,
        'execution_time': time_str,
    }


    os.makedirs(RESULTS_DIR, exist_ok=True)
    now = datetime.now()
    timestamp_str = now.strftime("%d%m%y_%H%M")
    dataset_base_name = selected_name.replace('.', '_')
    ml_scorer_name = type(ml_scorer).__name__
    pdf_filename = f"reporte_{dataset_base_name}_{ml_scorer_name}_{timestamp_str}.pdf"
    full_pdf_path = os.path.join(RESULTS_DIR, pdf_filename)

    generate_pdf_report(
        dataset_info, 
        chainer, 
        output_filename=full_pdf_path
    )

    print("\n-----------------------------------------------------")
    save_best_dataset = input(
        "¿Desea guardar el dataset con el MEJOR score (X_train y y_train)? (s/n): "
    ).lower().strip() == 's'


    if save_best_dataset:
        # El chainer debe tener los datos con el mejor accuracy
        X_best = chainer.best_X_train # <-- Asumiendo que guardaste el mejor X_train
        y_best = chainer.y_train 
        
        # Convertir X_best a DataFrame (si es un NumPy array)
        if isinstance(X_best, np.ndarray):
            X_best_df = pd.DataFrame(X_best)
        else:
            X_best_df = X_best
            
        # Unir X y y para un solo CSV
        X_best_df['target'] = y_best.values if isinstance(y_best, pd.Series) else y_best 

        csv_filename = f"best_dataset_{dataset_base_name}_{ml_scorer_name}_{timestamp_str}.csv"
        full_csv_path = os.path.join(RESULTS_DIR, csv_filename)
        
        X_best_df.to_csv(full_csv_path, index=False)
        
        print(f"Dataset guardado exitosamente en: {os.path.abspath(full_csv_path)}")

    
    # 2. Mostrar resultados finales
    print("\n\n#####################################################")
    print("         RESULTADOS FINALES DEL ENCADENAMIENTO         ")
    print("#####################################################")
    
    print(f"Accuracy Inicial (Base): {chainer.initial_accuracy:.4f}")

    if final_results:
        for result in final_results:
            print(f"Paso {result['step']}: FE '{result['fe_added']}' agregado. -> Accuracy: {result['new_accuracy']:.4f} | Tamaño: {result['new_shape']}")
        
        print(f"\nMEJOR ACCURACY FINAL: {chainer.best_accuracy:.4f}")
    else:
        print("No se encontró ningún FE que mejorara el Accuracy Base.")
        print(f"MEJOR ACCURACY FINAL: {chainer.best_accuracy:.4f}")
    
    print("#####################################################")

if __name__ == "__main__":
    main()

