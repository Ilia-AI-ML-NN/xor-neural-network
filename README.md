
# XOR Neural Network (from scratch, NumPy)

**RU | EN ниже**

## Описание (RU)
Мини-проект: нейронная сеть с одним скрытым слоем для решения задачи XOR.
Реализация *с нуля* на NumPy: прямой и обратный проход, градиентный спуск,
метрики (MSE, accuracy), визуализация кривых обучения и решающей границы, сохранение модели.

**Фичи**
- 2 входа → скрытый слой (настраиваемый размер) → 1 выход
- Сигмоида как функция активации
- Обучение градиентным спуском
- Accuracy + MSE
- Графики обучения (loss/accuracy) и решающая граница
- Сохранение весов в `xor_model.npz`

**Запуск**
```bash
git clone <your-repo-url>
cd xor-neural-network
pip install -r requirements.txt
python xor_nn.py
```

**Результат**
- В консоли: вероятности, бинарные предсказания и финальная точность
- Откроются 3 графика: loss, accuracy, решающая граница

---

## Description (EN)
A tiny neural network (1 hidden layer) that solves the XOR problem.
Implemented **from scratch** with NumPy: forward/backward pass, gradient descent,
metrics (MSE, accuracy), training curves & decision boundary visualization, model saving.

**Features**
- 2 inputs → hidden layer (configurable) → 1 output
- Sigmoid activation
- Gradient descent training
- Accuracy + MSE
- Training curves and decision boundary plots
- Save weights to `xor_model.npz`

**Usage**
```bash
git clone <your-repo-url>
cd xor-neural-network
pip install -r requirements.txt
python xor_nn.py
```

**Example output**
```
Final probabilities:
 [[0.01]
  [0.98]
  [0.99]
  [0.02]]
Final predictions:
 [[0]
  [1]
  [1]
  [0]]
Final Accuracy: 1.0
```

## License
MIT — see [LICENSE](LICENSE).
