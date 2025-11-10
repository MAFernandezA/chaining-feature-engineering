import os
import glob
import pandas as pd # type: ignore
from scipy.io import loadmat # type: ignore

def select_dataset_file(data_dir):
    """
    Lista los archivos .mat en el directorio 'data/' y permite al usuario seleccionar uno.
    Retorna la ruta completa del archivo seleccionado y su nombre.
    """
    all_files = glob.glob(os.path.join(data_dir, '*.mat')) + glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not all_files:
        raise FileNotFoundError(f"No se encontraron archivos en el directorio: {data_dir}")

    print("\nArchivos disponibles:")
    file_map = {i + 1: os.path.basename(f) for i, f in enumerate(all_files)}
    
    for i, name in file_map.items():
        print(f"  {i}. {name}")

    while True:
        try:
            choice = input("Seleccione el número del dataset a utilizar: ")
            index = int(choice)
            if index in file_map:
                selected_name = file_map[index]
                selected_path = all_files[index - 1]
                print(f"Seleccionado: {selected_name}")
                return selected_path, selected_name
            else:
                print("Número inválido. Por favor, intente de nuevo.")
        except ValueError:
            print("Entrada inválida. Debe ingresar un número.")


def load_mat_to_dataframe(file_path):
    """
    Carga un archivo .mat con estructura X/Y y lo convierte a DataFrames de Pandas.
    Retorna X (features) y y (labels) como arrays de NumPy o DataFrame/Series.
    """
    data = loadmat(file_path)
    
    # Lógica para detectar X e Y (usando tu lógica anterior)
    posibles_X = [k for k in data.keys() if 'x' in k.lower() or 'data' in k.lower()]
    posibles_y = [k for k in data.keys() if 'y' in k.lower() or 'label' in k.lower()]

    X_key = posibles_X[0] if posibles_X else None
    y_key = posibles_y[0] if posibles_y else None
    
    if X_key is None or y_key is None:
        raise ValueError("No se encontraron las claves de características o etiquetas (X/Y) en el archivo .mat")

    X = data[X_key]
    # Usamos .ravel() para asegurar que 'y' sea un vector 1D
    y = data[y_key].ravel()
    
    # Convertimos X a DataFrame para manejar mejor las features
    X_df = pd.DataFrame(X)

    return X_df, y


def load_csv_to_dataframe(file_path):
    """
    Carga un archivo .csv y pide al usuario identificar la columna de etiquetas.
    Retorna X (features) como DataFrame y y (labels) como array de NumPy.
    """
    print("\n=== Carga de Archivo CSV ===")
    df = pd.read_csv(file_path)
    
    # 1. Mostrar las columnas
    print("Columnas encontradas en el CSV:")
    for i, col in enumerate(df.columns):
        print(f"  {i}. {col}")

    while True:
        try:
            # 2. Pedir al usuario que seleccione la columna de etiquetas (Y)
            choice = input("Seleccione el NÚMERO o NOMBRE de la columna de etiquetas (Y): ")
            if choice.isdigit():
                y_col = df.columns[int(choice)]
            elif choice in df.columns:
                y_col = choice
            else:
                raise ValueError
            
            # 3. Separar X y Y
            y = df[y_col].values.ravel()
            X_df = df.drop(columns=[y_col])
            
            print(f"Columna de etiquetas seleccionada: '{y_col}'")
            return X_df, y
            
        except (ValueError, IndexError):
            print("Selección inválida. Por favor, ingrese un número de índice o el nombre exacto de la columna.")
