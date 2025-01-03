# project_lin

![image](https://github.com/user-attachments/assets/c4f8b9cf-c845-4850-b3a9-49f60bf3e433)

Project Lin использует алгоритмы глубокого обучения для анализа и синтеза речи. Основная цель проекта — разработать модель, которая способна воспроизводить аудиозаписи с помощью голоса, который был скопирован на этапе обучения. Это может быть полезно в различных сферах, например, для создания аудиокниг, в играх, анимации и даже в технологиях помощи для людей с нарушениями речи.

В проекте я реализовала несколько ключевых компонентов, которые подробно опишу ниже.

# Датасет

Для обучения нейросети я собрала качественный датасет, состоящий из множества аудиозаписей, где разные люди произносит различные фразы. Данные включают как оригинальные записи, так и дополнительные фразы, чтобы обеспечить разнообразие и полноту модели. Я уделила особое внимание качеству аудиозаписей, чтобы добиться наилучших результатов.

# Предобработка данных
<img width="393" alt="{AA1760E8-F673-4943-B08E-3CFAECE74284}" src="https://github.com/user-attachments/assets/536d8c24-5f73-468a-a9c7-61bb8cd06170" />

Перед тем как подать аудиофайлы на вход нейросети, я провела этап предобработки данных. Этот процесс включает:

- Преобразование формата: Все аудиофайлы были приведены к единому формату и частоте дискретизации для упрощения обработки.
- Извлечение признаков: Я использовала методы извлечения признаков, такие как MFCC (Mel-frequency cepstral coefficients) и спектрограммы, чтобы получить числовые представления звуковых сигналов. Эти признаки помогают нейросети лучше анализировать звук.
- Нормализация: Я также нормализовала амплитуду звука, чтобы избежать искажений при обучении.

# Модель нейросети
<img width="395" alt="{471D45EC-4421-4151-BC28-675B84AD8275}" src="https://github.com/user-attachments/assets/2f69ae8e-7a26-43f1-a7ec-cd7dbc6f3311" />

В моем проекте реализована рекуррентная нейронная сеть (RNN), которая отлично подходит для обработки звуковых сигналов. Основные компоненты модели включают:

- LSTM (Long Short-Term Memory) слои: Эти слои позволяют модели учитывать временные зависимости в аудиоданных и хорошо подходят для обработки последовательностей.
- Входные и выходные слои: Они отвечают за получение извлечённых признаков на входе и генерацию аудиосигнала на выходе. Я также использую механизм внимания (attention mechanism), чтобы улучшить качество синтезируемой речи.
- Обратная связь: Модель обучается с использованием обратного распространения ошибки, что помогает ей корректировать предсказания и повышать точность синтеза голосовых данных.


# Обучение модели

Я обучала модель на большом объёме данных, используя алгоритм оптимизации Adam для улучшения скорости и эффективности обучения. В процессе обучения я применяла различные методы регуляризации, чтобы предотвратить переобучение и улучшить обобщающую способность модели. Процесс обучения состоял из нескольких эпох, где каждая эпоха включала множество итераций над обучающей выборкой.

# Тестирование и оценка

После завершения обучения я проверяла модель на тестовом наборе данных, который не использовался в процессе обучения. Это позволяло оценить качество синтезированного звука и точность копирования голоса. Я проводила как качественные, так и количественные оценки, слушая выходные аудиозаписи и сравнивая их с оригинальными записям

![image](https://github.com/user-attachments/assets/a9866763-115e-402c-b79f-b92a78a32cdd)
