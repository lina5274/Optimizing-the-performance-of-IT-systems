import prometheus_client
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Инициализация метрик Prometheus
cpu_usage = prometheus_client.Gauge('cpu_usage', 'CPU usage percentage')
memory_usage = prometheus_client.Gauge('memory_usage', 'Memory usage percentage')


# Функция сбора метрик
def collect_metrics():
    cpu_usage.set(get_cpu_usage())
    memory_usage.set(get_memory_usage())


# Функция оптимизации ресурсов
def optimize_resources(cpu, memory):
    if cpu > 80:
        scale_up_instance()
    elif cpu < 20:
        scale_down_instance()

    if memory > 90:
        clear_cache()


# Функция прогнозирования нагрузки
def predict_load(history_data):
    X = np.array([d['timestamp'] for d in history_data]).reshape(-1, 1)
    y = np.array([d['load'] for d in history_data])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    future_timestamps = np.array([time.time() + i * 3600 for i in range(24)]).reshape(-1, 1)
    predicted_load = model.predict(future_timestamps)

    return predicted_load


# Основной цикл работы системы
while True:
    collect_metrics()
    current_cpu = cpu_usage.get()
    current_memory = memory_usage.get()

    optimize_resources(current_cpu, current_memory)

    history_data = fetch_history_load_data()
    predicted_load = predict_load(history_data)

    schedule_optimizations(predicted_load)

    time.sleep(300)  # Проверка каждые 5 минут
