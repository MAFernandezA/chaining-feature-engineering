Versión Python necesaria para la instalación del requirements.txt: Python 3.11


#### PASOS PARA EJECUTAR HERRAMIENTA ####
py -0 
# Para revisar versiones instaladas

py -3.11 -m venv .venv 
# Crear entorno virtual con py 3.11

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process 
# Para ejecutar scripts locales sin firmar 

.\.venv\Scripts\activate 
# Activar entorno virtual

#### Instalar dependencias necesarias (Unica vez) ####
python.exe -m pip install --upgrade pip

python -m pip install -r requirements.txt 

python -m src.main # Ejecutar herramienta
