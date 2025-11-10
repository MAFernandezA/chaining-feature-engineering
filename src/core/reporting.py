from fpdf import FPDF 
import matplotlib.pyplot as plt 
import pandas as pd
import io 
import os 



def plot_chaining_accuracy(chain_results, initial_accuracy, dataset_name):
    """Genera la gráfica de línea del Accuracy a lo largo del encadenamiento."""
    
    # 1. Preparar los datos
    steps = [0] + [r['step'] for r in chain_results]
    accuracies = [initial_accuracy] + [r['new_accuracy'] for r in chain_results]
    
    # 2. Crear la gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(steps, accuracies, marker='o', linestyle='-', color='b')
    
    # Marcar el punto base y el mejor punto
    plt.scatter(0, initial_accuracy, color='green', marker='D', label=f'Base ({initial_accuracy:.4f})', zorder=5)
    
    # Encuentra el mejor accuracy global en la cadena
    if accuracies:
        best_acc = max(accuracies)
        best_step = steps[accuracies.index(best_acc)]
        plt.scatter(best_step, best_acc, color='red', marker='*', s=200, label=f'Mejor ({best_acc:.4f})', zorder=6)

    plt.title(f'Progreso del Accuracy en el Encadenamiento de FEs ({dataset_name})')
    plt.xlabel('Paso de Encadenamiento (0=Base)')
    plt.ylabel('Accuracy de Evaluación')
    plt.xticks(steps)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # 3. Guardar en buffer de memoria (para FPDF)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    plt.close() # Cerrar la figura para liberar memoria
    
    return img_buffer

def get_sample_text(X, max_rows=5, max_cols=10):
    """Retorna una muestra (head) del dataset como cadena de texto."""
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X)
    else:
        X_df = X
        
    shape_str = f"Tamaño: {X_df.shape}\n"
    
    if X_df.shape[1] > max_cols:
        shape_str += f"(Mostrando primeras {max_cols} columnas)\n"
        X_display = X_df.iloc[:, :max_cols]
    else:
        X_display = X_df

    # Usar to_string para una representación tabular clara
    sample_text = X_display.head(max_rows).to_string(max_rows=max_rows, max_cols=max_cols, index=False)
    
    return shape_str + sample_text

def generate_pdf_report(dataset_info, fe_chain_instance, output_filename="reporte_fe.pdf"):
    """
    Genera el archivo PDF con todas las estadísticas del proceso de encadenamiento.
    """
    
    # 1. Recolectar datos del objeto FeatureEngineeringChainer
    initial_accuracy = fe_chain_instance.initial_accuracy
    chain_results = fe_chain_instance.chain_results
    ml_scorer_name = type(fe_chain_instance.ml_scorer).__name__
    
    # Recolectar datos del dataset_info (que debes pasar desde main.py)
    dataset_name = dataset_info['name']
    original_shape = dataset_info['original_shape']
    pca_base_applied = dataset_info['pca_base_applied']
    base_dataset_shape = dataset_info.get('base_dataset_shape', fe_chain_instance.current_X_train.shape)
    base_X_train = dataset_info['base_X_train']    
    execution_time = dataset_info['execution_time']
    pca_threshold = fe_chain_instance.pca_threshold
    
    # 2. Inicializar FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- TÍTULO ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ESTADISTICAS CHAINING FEATURE ENGINEERING", 0, 1, "C")
    pdf.ln(5)

    # --- INFORMACIÓN GENERAL ---
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 7, f"Nombre del dataset: {dataset_name}", 0, 1)
    pdf.cell(0, 7, f"Tamaño original: {original_shape}", 0, 1)
    pdf.cell(0, 7, f"Modelo evaluador seleccionado: {ml_scorer_name}", 0, 1)
    pdf.cell(0, 7, f"Tiempo de ejecución total: {execution_time}", 0, 1)
    
    # --- PCA INFO ---
    pca_base = 'Sí' if dataset_info.get('pca_base_applied') else 'No'
    pca_dynamic = 'Sí' if fe_chain_instance.apply_pca_dynamic else 'No'
    pdf.cell(0, 7, f"PCA en dataset base: {pca_base}", 0, 1)
    pdf.cell(0, 7, f"PCA dinámico en encadenamiento: {pca_dynamic}", 0, 1)
    
    if pca_dynamic == 'Sí':
        pdf.cell(0, 7, f"Tamaño (features) para aplicar PCA dinámico: {pca_threshold}", 0, 1)

    if pca_base == 'Sí':
        pdf.cell(0, 7, f"Tamaño después de PCA base: {base_dataset_shape}", 0, 1)
    pdf.ln(5)

    # --- Accuracy BASE ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "PROCESO DE CHAINING FEATURE ENGINEERING", 0, 1)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 7, f"Accuracy BASE (Sin FE): {initial_accuracy:.4f}", 0, 1)
    
    # Muestra del dataset base (antes de cualquier FE)
    pdf.set_font("Courier", "", 8)
    pdf.multi_cell(0, 4, get_sample_text(base_X_train, max_rows=5, max_cols=10))
    pdf.ln(5)
    
    # --- CICLOS Y RESULTADOS ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 7, "RESULTADOS DE CADA PASO DE ENCADENAMIENTO", 0, 1)
    
    # Reconstrucción de la lógica de encadenamiento para el reporte
    # (Necesitarías guardar los resultados temporales de cada FE intentado, 
    # pero para simplificar, usaremos chain_results)
    
    pdf.set_font("Arial", "", 10)
    for step_data in chain_results:
        pdf.ln(2)
        pdf.set_fill_color(220, 220, 220)
        pdf.set_font("Arial", "B", 10)
        # Aquí se usa la info final del encadenamiento (lo que fue seleccionado)
        pdf.cell(0, 6, f"PASO {step_data['step']}: FE '{step_data['fe_added']}' seleccionado", 1, 1, 'L', True)
        
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 5, f"Accuracy final del paso: {step_data['new_accuracy']:.4f}", 0, 1)
        pdf.cell(0, 5, f"Tamaño del Dataset: {step_data['new_shape']}", 0, 1)
        
        # Opcional: Mostrar la muestra del dataset en este paso (requiere guardar X_train_new)
        # Esto es complejo, ya que X_train_new no se guarda en chain_results.
        # Por ahora, solo reportaremos la forma.

        if 'sample_X_train' in step_data:
            pdf.set_font("Courier", "", 8)
            pdf.multi_cell(0, 4, get_sample_text(step_data['sample_X_train'], max_rows=5, max_cols=10))
            
        pdf.ln(2)

    # --- GRÁFICO DE LÍNEA ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "GRÁFICA DE PROGRESO DE ACCURACY", 0, 1, "C")
    pdf.ln(5)

    # Generar la gráfica y obtener el buffer de memoria
    img_buffer = plot_chaining_accuracy(chain_results, initial_accuracy, dataset_name)
    
    # Incrustar la imagen del gráfico
    pdf.image(img_buffer, x=20, y=pdf.get_y(), w=170)


    # 3. Guardar el archivo final
    pdf.output(output_filename)
    print(f"\n🎉 ¡Reporte PDF generado exitosamente en: {os.path.abspath(output_filename)}!")