import pandas as pd 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split 



def apply_pca_reduction(X, n_components=0.95, random_state=42):
    """
    Aplica PCA para reducir la dimensionalidad.
    'n_components=0.95' significa mantener el 95% de la varianza.
    """
    print("... Aplicando Reducción de Dimensionalidad (PCA) ...")
    pca = PCA(n_components=n_components, random_state=random_state)
    
    # X_reduced será un array NumPy
    X_reduced = pca.fit_transform(X)
    
    return X_reduced


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Divide el dataset X y las etiquetas y en conjuntos de entrenamiento y prueba.
    """
    print("Dividiendo Dataset (Train/Test)")
    
    # Realizar la división
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Stratify=y es importante para clasificación si las clases están desbalanceadas
    
    print(f"Dataset dividido correctamente:")
    print(f"  Entrenamiento (X_train): {X_train.shape}, (y_train): {y_train.shape}")
    print(f"  Prueba (X_test): {X_test.shape}, (y_test): {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def print_dataset_sample(X, fe_name, max_rows=5, max_cols=10):
    """Imprime una muestra (head) del nuevo dataset y su forma."""
    
    # 1. Convertir a DataFrame si es necesario (para usar .head() y formato)
    if not isinstance(X, pd.DataFrame):
        # Tomar una muestra del array de numpy
        X_df = pd.DataFrame(X)
    else:
        X_df = X
        
    print(f"\n[Dataset {fe_name} Info]")
    print(f"  Tamaño (Filas, Columnas): {X_df.shape}")
    
    # 2. Manejo de columnas excedentes
    if X_df.shape[1] > max_cols:
        print(f"  Mostrando un máximo de {max_cols} columnas de las {X_df.shape[1]} totales...")
        X_display = X_df.iloc[:, :max_cols]
    else:
        X_display = X_df

    # 3. Imprimir muestra
    # Usamos to_string para un mejor control de la salida en consola
    print("--- MUESTRA X_train (HEAD) ---")
    print(X_display.head(max_rows).to_string(max_rows=max_rows, max_cols=max_cols, index=False))
    print("------------------------------")

