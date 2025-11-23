# LAB 4

## Baseline Model
**Struktura**:
- model.add(normalizer)
- model.add(layers.Dense(32, activation="relu"))
- model.add(layers.Dense(3, activation="softmax"))

**Parametry treningowe**:
- Optimizer: Adam(learning_rate=0.001)
- Epochs: 50
- Batch size: 16

**Wynik na serwerze**:
- Test Accuracy: 0.9722
- Test Loss: 0.1258

**Wynik lokalny**:
- Test Accuracy: 0.9444
- Test Loss: 0.0806 

## Tuning hiperparametrów:

**Optymalizowane parametry**:
- Liczba neuronów(units): 8-64
- Dropout: 0.0 - 0.6, step co 0.1
- Learning rate: 1e-4 - 1e-2

Do tuningu użyto metody RandomSearch, ilość prób ustawiona na 10.

**Najlepsze hiperparametry(serwer)**:
- Ilość neuronów w warstwie ukrytej: 32
- Poziom dropoutu: 0.1
- Learning rate: 0.006108123523092101

**Najlepsze hiperparametry(lokalnie)**:
- Ilość neuronów w warstwie ukrytej: 16
- Poziom dropoutu: 0.30000000000000004
- Learning rate: 0.004859430383732818

## Model po tuningu

**Wynik na serwerze**:
- Test Accuracy: 1.0000
- Test Loss: 0.0097 

**Wynik lokalny**:
- Test Accuracy: 1.0000
- Test Loss: 0.0254


## Macierz pomyłek
```bash
[[10  0  0]
 [ 0 17  0]
 [ 0  0  9]]
```

Model poprawnie sklasyfikował wszystkie próbki walidacyjne.

## Wnioski
- Normalizacja wyników przyniosła lepszy wynik niż korzystanie z StandardScalera
- Optymalizowany model wykazał lepszą dokładność od baseline modelu sięgając perfect score
- Tuner postawił na liczbę neuronów 32, czyli tyle ile ma domyślnie model bazowy, co ciekawe lokalnie, testowany na cpu wybrał opcję mniejszą - 16 neuronów. Na obu osiągnał jednak dobry wynik, co może sugerować, że 16 może być wystarczająca liczbą.
- Dataset jest bardzo mały, więc osiągnięcie takich wyników nie jest dużym problemem.