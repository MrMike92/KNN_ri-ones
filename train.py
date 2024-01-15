"""MIT License
Copyright (c) 2024 MrMike92
Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y de los archivos de documentación asociados (el "Software"), para tratar el Software sin restricciones, incluyendo, sin limitación, los derechos de uso, copia, modificación, fusión, publicación, distribución, sublicencia y/o venta de copias del Software, y para permitir a las personas a las que se les proporcione el Software a hacerlo, sujeto a las siguientes condiciones:
El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias o porciones sustanciales del Software.
EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA, INCLUYENDO PERO NO LIMITADO A LAS GARANTÍAS DE COMERCIABILIDAD, ADECUACIÓN PARA UN PROPÓSITO PARTICULAR Y NO INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DE LOS DERECHOS DE AUTOR SERÁN RESPONSABLES POR CUALQUIER RECLAMO, DAÑO U OTRA RESPONSABILIDAD, YA SEA EN UNA ACCIÓN DE CONTRATO, AGRAVIO O DE OTRO MODO, DERIVADA DE, FUERA DE O EN CONEXIÓN CON EL SOFTWARE O EL USO U OTROS TRATOS EN EL SOFTWARE."""

import numpy as np
from PIL import Image
import os
import csv

class imagen:
    def __init__(self, ruta_imagen):
        self.ruta_imagen = ruta_imagen
    
    def imagenEscalaGrises(self, nuevo_ancho, nuevo_alto):
        imagen = Image.open(self.ruta_imagen)
        imagen_gris = imagen.convert("L")
        
        # Redimensiona a 64x64 la imagen en escala de grises
        imagen_gris_redimensionada = imagen_gris.resize((nuevo_ancho, nuevo_alto))
        
        # Calcular los valores estadísticos
        resultados = self.obtener_valores_estadisticos(imagen_gris_redimensionada)

        # Obtener la etiqueta según la carpeta de origen
        etiqueta = 1 if "stone" in self.ruta_imagen else 0

        # Agregar la etiqueta a la lista de resultados
        resultados.append(etiqueta)
        
        return resultados

    def obtener_valores_estadisticos(self, imagen_gris_redimensionada):
        matriz = list(imagen_gris_redimensionada.getdata())

        # Obtener las dimensiones de la matriz
        dimensiones = imagen_gris_redimensionada.size[::-1]
        filas, columnas = dimensiones

        # Definir el tamaño del kernel
        tamanio_kernel = 3

        # Calcular el número de pasos para recorrer la matriz con el kernel
        pasos_filas = filas // tamanio_kernel
        pasos_columnas = columnas // tamanio_kernel

        # Crear una lista para almacenar los resultados de los valores estadísticos
        resultados = []

        # Recorrer la matriz con el kernel
        for i in range(0, pasos_filas, tamanio_kernel):
            for j in range(0, pasos_columnas, tamanio_kernel):
                # Obtener la región del kernel en la matriz
                region_kernel = []
                for k in range(tamanio_kernel):
                    fila = matriz[(i + k) * columnas + j:(i + k) * columnas + j + tamanio_kernel]
                    region_kernel.extend(fila)

                # Calcular los valores estadísticos de la región del kernel
                valor_minimo = min(region_kernel)
                valor_maximo = max(region_kernel)
                media = np.mean(region_kernel)
                desviacion_estandar = np.std(region_kernel)
                varianza = np.var(region_kernel)
                energia = np.sum(np.array(region_kernel)**2)
                entropia = -np.sum(np.array(region_kernel) * np.log2(np.array(region_kernel) + np.finfo(float).eps))

                # Agregar los valores a la lista de resultados
                resultados.extend([valor_minimo, valor_maximo, media, desviacion_estandar, varianza, energia, entropia])

        return resultados

def main():
    normal_folder = r"normal"
    stone_folder = r"stone"
    resultados_totales = []

    # Procesar las imágenes de la carpeta "normal"
    nombres_archivos_normal = os.listdir(normal_folder)
    for nombre_imagen in nombres_archivos_normal:
        ruta_imagen = os.path.join(normal_folder, nombre_imagen)
        normal_instance = imagen(ruta_imagen)
        resultados = normal_instance.imagenEscalaGrises(64, 64)
        resultados_totales.append(resultados)

    # Procesar las imágenes de la carpeta "stone"
    nombres_archivos_stone = os.listdir(stone_folder)
    for nombre_imagen in nombres_archivos_stone:
        ruta_imagen = os.path.join(stone_folder, nombre_imagen)
        stone_instance = imagen(ruta_imagen)
        resultados = stone_instance.imagenEscalaGrises(64, 64)
        resultados_totales.append(resultados)

    # Escribir los resultados en un archivo CSV
    ruta_csv = "train.csv"

    with open(ruta_csv, mode='w', newline='') as archivo_csv:
        writer = csv.writer(archivo_csv)

        ''' # Escribir la cabecera del archivo CSV
        cabecera = ["Mínimo", "Máximo", "Media", "Desviación Estándar", "Varianza", "Energía", "Entropía", "Etiqueta"]
        writer.writerow(cabecera)'''

        # Escribir los datos de cada imagen en una fila del archivo CSV
        for resultados in resultados_totales:
            writer.writerow(resultados)

if __name__ == '__main__':
    main()
    print("Programa Terminado...")