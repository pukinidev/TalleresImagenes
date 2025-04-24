# Análisis y Discusión de Resultados

## Modelos Entrenados:

Modelo 1:

```python
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(16, activation='relu'),
    Dropout(0.2),
    Dense(len(categories), activation='softmax')
])
model.summary()



              precision    recall  f1-score   support

      Normal       0.96      0.96      0.96        82
    Covid-19       0.96      0.96      0.96        85

    accuracy                           0.96       167
   macro avg       0.96      0.96      0.96       167
weighted avg       0.96      0.96      0.96       167

```


Modelo 2:

```python
class CNNHyperModel(HyperModel):
    
    def build(self, hp):
        
        model = Sequential()
        
        model.add(layers.Conv2D(
            filters=hp.Int('filters_1', min_value=16, max_value=32, step=16),
            kernel_size=hp.Choice('kernel_size_1', values=[3]),
            activation='relu',
            input_shape=(128, 128, 3)
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        
    
        model.add(layers.Conv2D(
            filters=hp.Int('filters_2', min_value=32, max_value=64, step=32),
            kernel_size=hp.Choice('kernel_size_2', values=[3]),
            activation='relu'
        ))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(
            units=hp.Int('dense_units', min_value=32, max_value=64, step=32),
            activation='relu'
        ))
        model.add(layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.2, max_value=0.3, step=0.1)))
        model.add(layers.Dense(len(categories), activation='softmax'))

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model


              precision    recall  f1-score   support

      Normal       0.99      0.96      0.98        82
    Covid-19       0.97      0.99      0.98        85

    accuracy                           0.98       167
   macro avg       0.98      0.98      0.98       167
weighted avg       0.98      0.98      0.98       167


```

Modelo 3:

```python

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),  
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),   
    Dropout(0.3),              
    Dense(len(categories), activation='softmax')
])


              precision    recall  f1-score   support

    Covid-19       0.96      0.99      0.98        83
      Normal       0.99      0.96      0.98        84

    accuracy                           0.98       167
   macro avg       0.98      0.98      0.98       167
weighted avg       0.98      0.98      0.98  

```



A partir del análisis comparativo entre los tres modelos entrenados, se observa una mejora progresiva en el rendimiento al incrementar la profundidad de la red y ajustar adecuadamente los hiperparámetros. El Modelo 1, con una arquitectura más simple y menor regularización, logra una precisión del 96%, lo cual es bastante bueno, pero ligeramente inferior a los modelos posteriores. El Modelo 2 introduce una búsqueda de hiperparámetros con `HyperModel`, lo que permite explorar combinaciones óptimas de filtros, unidades densas y tasas de *dropout*, alcanzando una precisión del 98%. Esto sugiere que una arquitectura más flexible y adaptativa contribuye significativamente a mejorar la capacidad de aprendizaje. Por su parte, el Modelo 3, aunque no utiliza búsqueda automática, incorpora una capa convolucional adicional (64 filtros) y aumenta tanto el número de neuronas en la capa densa como el *dropout* a 0.3. Estos cambios refuerzan la capacidad del modelo para extraer características relevantes y evitar el sobreajuste, logrando una precisión equivalente al Modelo 2. En conjunto, los resultados evidencian que una mayor profundidad y una adecuada regularización mejoran tanto la capacidad de aprendizaje como la generalización del modelo. Futuras mejoras podrían incluir el uso de *batch normalization*, técnicas de aumento de datos o arquitecturas preentrenadas para potenciar aún más el desempeño.