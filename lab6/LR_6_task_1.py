import numpy as np
from numpy.random import randn

class RNN:
  # Класична рекурентна нейронна мережа

  def __init__(self, input_size, output_size, hidden_size=64):
    # Weights
    self.Whh = randn(hidden_size, hidden_size) / 1000
    self.Wxh = randn(hidden_size, input_size) / 1000
    self.Why = randn(output_size, hidden_size) / 1000

    # Зміщення
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

  def forward(self, inputs):
    '''
    Виконання передачі нейронної мережі за допомогою вхідних даних
    Повернення результатів виведення та прихованого стану
    Вивід – це масив одного унітарного вектора з формою (input_size, 1)
    '''
    h = np.zeros((self.Whh.shape[0], 1))

    self.last_inputs = inputs
    self.last_hs = { 0: h }

    # Виконання кожного кроку в нейронній мережі RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      self.last_hs[i + 1] = h

    # Compute the output
    y = self.Why @ h + self.by

    return y, h

  def backprop(self, d_y, learn_rate=2e-2):
    '''
    Виконання фази зворотного розповсюдження мережі RNN.
    - d_y (dL/dy) має форму (output_size, 1).
    - learn_rate є дійсним числом float.
    '''
    n = len(self.last_inputs)

    # Підрахунок dL/dWhy та dL/dby.
    d_Why = d_y @ self.last_hs[n].T
    d_by = d_y

    # Ініціалізація dL/dWhh, dL/dWxh, і dL/dbh до нуля
    d_Whh = np.zeros(self.Whh.shape)
    d_Wxh = np.zeros(self.Wxh.shape)
    d_bh = np.zeros(self.bh.shape)

    # Обчислення dL/dh для останнього h.
    # dL/dh = dL/dy * dy/dh
    d_h = self.Why.T @ d_y

    # Зворотне розповсюдження по часу.
    for t in reversed(range(n)):
      # Среднее значение: dL/dh * (1 - h^2)
      temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

      # dL/db = dL/dh * (1 - h^2)
      d_bh += temp

      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
      d_Whh += temp @ self.last_hs[t].T

      # dL/dWxh = dL/dh * (1 - h^2) * x
      d_Wxh += temp @ self.last_inputs[t].T

      # Далі dL/dh = dL/dh * (1 - h^2) * Whh
      d_h = self.Whh @ temp

    # Відсікаємо, щоб попередити розрив градієнтів.
    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
      np.clip(d, -1, 1, out=d)

    # Обновляємо ваги і зміщення з використанням градієнтного спуску.
    self.Whh -= learn_rate * d_Whh
    self.Wxh -= learn_rate * d_Wxh
    self.Why -= learn_rate * d_Why
    self.bh -= learn_rate * d_bh
    self.by -= learn_rate * d_by
