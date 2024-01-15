# k-NN de detección de piedras en el riñon
Un clasificador KNN para una base de datos de imágenes de resonancia magnética de los riñones para la detección de piedras

## Instrucciones de uso.

- Clona este repositorio en tu máquina local.
- Asegure que el respositorio se haya descargado correctamente.
- Abre el programa que deseas ejecutar en tu entorno de desarrollo que soporte Python 3.11.2 64-bit.
- Asegure de que cuando se ejecute el programa, los archivos para el entrenamiento (train.csv) y de prueba (test.csv) esten en el lugar donde se encuentre el programa (knn_validacion_cruzada.py)

## Funcionamiento
Para este caso primero se tiene que pasar las imagenes a un archivo con valores separados por comas (archivo cvs) para procesar los datos y clasificar las imagenes si son riñones sanos o con piedras, por lo que se sugiere realizar los siguentes pasos para ejecutar el programa correctamente con cualquier otra base de datos (ya sea con imagenes a color o blanco y negro o escala de grises):

<br> 1. Descargar la base de datos.

> [!IMPORTANT]
> La base de datos de imangenes utilizada para este proyecto pertenece a su resprectivo creador, GONZAJL.
> <br><br>Link de la base de datos de las imagenes: https://www.kaggle.com/datasets/gonzajl/riones-cyst-stone-tumor-normal-dataset?rvi=1

<br> 2. Mantener las carpetas "Stone" y "Normal"

<br> 3. Ejecutar el programa train.py para convertir las dos carpetas de imagenes "Stone" (los que tienen los riñones con calculos renales o piedras) y "Normal" (los que tienen los riñones sanos) a un archivo train.csv

> [!WARNING]
> Si hay problemas en ejecutar el programa pruebas_imagenes.py, se debe de modificar la linea #13 y reemplazarla con la extención de las imagenes que se desea trabajar (por ejemplo: .png, .jpeg, .jpg, etc.), támbien puedes modificar la extensión en la que se guardaran las imagenes en la carpeta test en la linea #16
> ![image](https://github.com/MrMike92/KNN_tumores_cerebro/assets/93272523/d58851fd-43b7-4b83-892b-1db73a91a566)

<br> 4. Ejecutar knn_validacion_cruzada.py para realizar la clasificación de las imagenes.

> [!NOTE]
> Siéntete libre de personalizar el código, ya sea cambiando el valor de k (número de vecinos más cercanos) o cambiando la distancia euclidiana por otra distancia para ver si se pueda mejorar la precisión del clasificador.
> <br><br>Este código no tiene dependencias externas y debería funcionar con cualquier entorno Python 3.x.

Si deseas contribuir a este proyecto, puedes enviar solicitudes de extracción (pull requests) con mejoras o características adicionales y si tienes alguna pregunta o problema, puedes contactarme a través de mi perfil de GitHub MrMike92, en un futuro planeo abrir un correo para poder contactarme. 🐢
