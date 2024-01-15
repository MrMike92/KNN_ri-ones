"""MIT License
Copyright (c) 2024 MrMike92
Se concede permiso, de forma gratuita, a cualquier persona que obtenga una copia de este software y de los archivos de documentación asociados (el "Software"), para tratar el Software sin restricciones, incluyendo, sin limitación, los derechos de uso, copia, modificación, fusión, publicación, distribución, sublicencia y/o venta de copias del Software, y para permitir a las personas a las que se les proporcione el Software a hacerlo, sujeto a las siguientes condiciones:
El aviso de copyright anterior y este aviso de permiso se incluirán en todas las copias o porciones sustanciales del Software.
EL SOFTWARE SE PROPORCIONA "TAL CUAL", SIN GARANTÍA DE NINGÚN TIPO, EXPRESA O IMPLÍCITA, INCLUYENDO PERO NO LIMITADO A LAS GARANTÍAS DE COMERCIABILIDAD, ADECUACIÓN PARA UN PROPÓSITO PARTICULAR Y NO INFRACCIÓN. EN NINGÚN CASO LOS AUTORES O TITULARES DE LOS DERECHOS DE AUTOR SERÁN RESPONSABLES POR CUALQUIER RECLAMO, DAÑO U OTRA RESPONSABILIDAD, YA SEA EN UNA ACCIÓN DE CONTRATO, AGRAVIO O DE OTRO MODO, DERIVADA DE, FUERA DE O EN CONEXIÓN CON EL SOFTWARE O EL USO U OTROS TRATOS EN EL SOFTWARE."""

import numpy as np
import math
import random
import matplotlib.pyplot as plt

class clasificador_KNN:
    def __init__(self, k):
        self.k = k
        self.predicciones_totales = []

    def load_data(self, archivo_entrenamiento):
        set_entrenamiento = self.load_dataset(archivo_entrenamiento)
        return set_entrenamiento

    def load_dataset(self, archivo):
        dataset = []
        with open(archivo, 'r') as file:
            for linea in file:
                instancia = linea.strip().split(',')
                instancia = [float(valor) for valor in instancia]
                dataset.append(instancia)

        random.shuffle(dataset)
        return np.array(dataset)

    def distancia_euclidiana(self, instancia1, instancia2):
        distancia = np.sum((instancia1[:-1] - instancia2[:-1]) ** 2)
        return math.sqrt(distancia)

    def calcular_vecinos(self, set_entrenamiento, instancia_prueba):
        distancias = []
        for instancia_entrenamiento in set_entrenamiento:
            dist = self.distancia_euclidiana(instancia_prueba, instancia_entrenamiento)
            distancias.append((instancia_entrenamiento, dist))
        distancias.sort(key=lambda x: x[1])
        vecinos = np.array([item[0] for item in distancias[:self.k]])
        return vecinos

    def voto_mayoria(self, vecinos):
        etiquetas, votos = np.unique(vecinos[:, -1], return_counts=True)
        etiqueta_predecida = etiquetas[np.argmax(votos)]
        return etiqueta_predecida

    def predicciones(self, set_entrenamiento, archivo_predicciones, archivo_predicciones_correctas, archivo_predicciones_incorrectas):
        predicciones_correctas = []
        predicciones_incorrectas = []

        with open(archivo_predicciones, 'w') as archivo_total, open(archivo_predicciones_correctas, 'w') as archivo_correcto, open(archivo_predicciones_incorrectas, 'w') as archivo_incorrecto:
            for i, instancia_prueba in enumerate(set_entrenamiento):
                vecinos = self.calcular_vecinos(set_entrenamiento, instancia_prueba)
                voto_mayoria = self.voto_mayoria(vecinos)
                self.predicciones_totales.append(voto_mayoria)
                etiqueta_real = instancia_prueba[-1]
                linea = f"Fila {i+1}: Etiqueta real: {etiqueta_real}, Etiqueta predecida: {voto_mayoria}\n"
                archivo_total.write(linea)

                if voto_mayoria == etiqueta_real:
                    predicciones_correctas.append(voto_mayoria)
                    archivo_correcto.write(linea)
                else:
                    predicciones_incorrectas.append(voto_mayoria)
                    archivo_incorrecto.write(linea)

            # Calcular porcentajes y guardar en archivos
            total_samples = len(set_entrenamiento)
            accuracy = (len(predicciones_correctas) / total_samples) * 100
            error_rate = (len(predicciones_incorrectas) / total_samples) * 100
            archivo_total.write(f"\nPorcentaje de predicciones correctas: {accuracy:.2f}%\n")
            archivo_total.write(f"Porcentaje de predicciones incorrectas: {error_rate:.2f}%\n")
            archivo_correcto.write(f"\nPorcentaje de predicciones correctas: {accuracy:.2f}%\n")
            archivo_incorrecto.write(f"\nPorcentaje de predicciones incorrectas: {error_rate:.2f}%\n")

        return accuracy

    def graficar_clasificacion(self, set_entrenamiento, fold):
        # Calcular el promedio de los valores de cada fila
        promedios = np.mean(set_entrenamiento[:, :-1], axis=1)

        # Obtener los valores de X (promedio de los valores de cada fila)
        x = promedios

        # Obtener las etiquetas reales
        y_real = set_entrenamiento[:, -1]

        # Obtener las etiquetas predichas
        y_pred = np.array(self.predicciones_totales)[:len(set_entrenamiento)]

        # Graficar los puntos, coloreados según la etiqueta real y la etiqueta predicha
        plt.scatter(x, y_real, c='blue', label='Etiqueta Real')
        plt.scatter(x, y_pred, c='red', marker='x', label='Etiqueta Predicha')
        plt.xlabel('Promedio de los valores de píxeles de cada instancia en el conjunto de entrenamiento')
        plt.ylabel('Etiqueta')
        plt.legend()
        plt.title(f'Clasificación KNN - Fold {fold + 1}')
        plt.show()

    def validacion_cruzada(self, archivo_entrenamiento, k_folds=5):
        set_entrenamiento = self.load_data(archivo_entrenamiento)
        fold_size = len(set_entrenamiento) // k_folds
        accuracies = []

        for i in range(k_folds):
            start = i * fold_size
            end = (i + 1) * fold_size
            set_entrenamiento_fold = np.concatenate([set_entrenamiento[:start], set_entrenamiento[end:]])
            set_prueba_fold = set_entrenamiento[start:end]
            archivo_predicciones = f'predicciones_fold_{i + 1}.txt'
            archivo_correctas = f'predicciones_correctas_fold_{i + 1}.txt'
            archivo_incorrectas = f'predicciones_incorrectas_fold_{i + 1}.txt'
            accuracy = self.predicciones(set_prueba_fold, archivo_predicciones, archivo_correctas, archivo_incorrectas)

            # Graficar clasificación para el fold actual
            self.graficar_clasificacion(set_prueba_fold, i)

            # Almacenar la precisión en la lista
            accuracies.append(accuracy)

            # Mostrar porcentaje para el fold actual
            print(f'Fold {i + 1}: Precisión: {accuracy:.2f}%')

        # Graficar porcentajes de precisión para cada fold
        plt.bar(range(1, k_folds + 1), accuracies)
        plt.xlabel('Fold')
        plt.ylabel('Precisión (%)')
        plt.title(f'Precisión por Fold en {k_folds}-fold cross-validation')
        plt.show()

        # Calcular la precisión promedio sobre todos los folds
        accuracy_promedio = np.mean(accuracies)
        print(f'Precisión promedio en {k_folds}-fold cross-validation: {accuracy_promedio:.2f}%')

def main():
    k = 1
    print("K =", k)
    clasificador = clasificador_KNN(k)
    archivo_entrenamiento = 'train.csv'
    clasificador.validacion_cruzada(archivo_entrenamiento)
    print("Programa Terminado...")

if __name__ == '__main__':
    main()