import numpy as np # type: ignore
from src.fe_models.autofeat_fe import AutoFeatTransformer 
from src.fe_models.polynomial_fe import PolynomialFeatTransformer
from src.fe_models.featuretools_fe import FeatureToolsTransformer
from src.fe_models.boruta_fe import BorutaFeatureSelector
from sklearn.decomposition import PCA
from src.fe_models.list_fe import get_list_available_fes
from src.core.utils import print_dataset_sample


class FeatureEngineeringChainer:
    def __init__(self, X_train, y_train, X_test, y_test, apply_pca_dynamic=False, pca_threshold=99999, ml_scorer=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.apply_pca_dynamic = apply_pca_dynamic
        self.pca_threshold = pca_threshold
        self.ml_scorer = ml_scorer
        self.pca_model = PCA(n_components=0.95, random_state=42)
        
        # Lista inicial de FEs disponibles (puedes añadir más aquí)
        # Se guarda como (nombre, instancia_de_la_clase)
        self.available_fes = get_list_available_fes()
        
        self.initial_accuracy = 0.0
        self.best_accuracy = 0.0
        self.current_X_train = X_train.copy()
        self.current_X_test = X_test.copy()
        self.chain_results = []
    
    def run_initial_evaluation(self):
        """Evalúa el dataset BASE (sin ningún FE aplicado)."""
        model = self.ml_scorer
        acc = model.train_and_evaluate(self.X_train, self.y_train, self.X_test, self.y_test)

        self.initial_accuracy = acc
        self.best_accuracy = acc
        print(f"\nAccuracy BASE (sin FE): {acc:.4f}")
        return acc



    def run_chaining(self):
        # 1. Evaluación Base
        self.run_initial_evaluation()
        
        cycle_count = 0
        
        # 2. Bucle principal: Repetir mientras haya FEs y la precisión mejore
        while self.available_fes:
            cycle_count += 1
            print("\n=============================================")
            print(f"CICLO {cycle_count}. Mejor Accuracy actual: {self.best_accuracy:.4f} | FEs restantes: {len(self.available_fes)}")
            print("=============================================")
            
            best_fe_in_cycle = None
            best_acc_in_cycle = -1.0
            
            # --- 3. Probar todos los FEs restantes ---
            
            # Copiamos la lista para poder modificarla sin interrumpir el ciclo for
            fes_to_test = list(self.available_fes) 
            
            for fe_name, fe_transformer in fes_to_test:
                print(f"\n-> Probando FE: {fe_name}...")
                
                # Aplicar el FE al dataset actual (que puede tener features de ciclos anteriores)
                try:
                    # fit_transform en X_train, luego transform en X_test
                    X_train_new = fe_transformer.fit_transform(self.current_X_train, self.y_train)
                    X_test_new = fe_transformer.transform(self.current_X_test)
                except Exception as e:
                    print(f"Error al aplicar {fe_name}. Saltando este FE. Error: {e}")
                    continue


                print_dataset_sample(X_train_new, fe_name)

                if self.apply_pca_dynamic:
                    current_n_features = X_train_new.shape[1]
                    
                    if current_n_features > self.pca_threshold:
                        print(f"\n[PCA Dinámico] Dimensionalidad ({current_n_features}) excede el umbral ({self.pca_threshold}). Aplicando PCA...")
                        
                        try:
                            # 1. Ajustar y transformar X_train
                            # Usamos el modelo PCA guardado en __init__
                            X_train_new = self.pca_model.fit_transform(X_train_new)
                            
                            # 2. Transformar X_test
                            X_test_new = self.pca_model.transform(X_test_new)
                            
                            print(f"PCA aplicado. Nuevo tamaño: {X_train_new.shape}")
                            
                        except Exception as e:
                            # Captura errores si el FE anterior dejó NaNs o Infs
                            print(f"Error al aplicar PCA dinámico. Continuado sin PCA. Error: {e}")

                # 4. Medir y evaluar el nuevo dataset
                model = self.ml_scorer
                current_acc = model.train_and_evaluate(X_train_new, self.y_train, X_test_new, self.y_test)
                
                print(f"   Accuracy con {fe_name}: {current_acc:.4f}")
                
                # 5. Verificar si es el mejor de este ciclo
                if current_acc > best_acc_in_cycle:
                    best_acc_in_cycle = current_acc
                    # Guardamos todos los elementos necesarios para el paso siguiente
                    best_fe_in_cycle = {
                        "name": fe_name, 
                        "transformer": fe_transformer, 
                        "X_train": X_train_new, 
                        "X_test": X_test_new, 
                        "accuracy": current_acc
                    }
            
            # --- 6. Tomar decisión (Encadenamiento) ---
            
            # Comparamos el mejor resultado del ciclo con el mejor global
            # and best_acc_in_cycle > self.best_accuracy: <- Agregar para parar el encadenamiento hasta que no se detecte mejora
            if best_fe_in_cycle :     
                result = best_fe_in_cycle

                is_global_best_flag = best_acc_in_cycle > self.best_accuracy
                
                if best_acc_in_cycle > self.best_accuracy:
                    print(f"\n¡Éxito! El FE '{result['name']}' mejoró el Accuracy de {self.best_accuracy:.4f} a {best_acc_in_cycle:.4f}")
                    self.best_accuracy = best_acc_in_cycle
                else:
                    print(f"\nEncadenando FE '{result['name']}' aunque el Accuracy haya cambiado de {self.best_accuracy:.4f} a {best_acc_in_cycle:.4f}")        

                # Actualizar el mejor Accuracy y el dataset base para la próxima iteración
                self.current_X_train = result['X_train']
                self.current_X_test = result['X_test']
                
                self.available_fes = [(n, t) for n, t in self.available_fes if n != result['name']]

                self.chain_results.append({
                    "step": len(self.chain_results) + 1,
                    "fe_added": result['name'],
                    "new_accuracy": best_acc_in_cycle,
                    "new_shape": self.current_X_train.shape,
                    "is_global_best": is_global_best_flag
                })
                
            else:
                # La condición de parada: el accuracy no mejora o empieza a bajar
                print(f"\nNo hubo mejora en el Accuracy (Mejor global: {self.best_accuracy:.4f}). Deteniendo el encadenamiento.")
                break 

        return self.chain_results
    
