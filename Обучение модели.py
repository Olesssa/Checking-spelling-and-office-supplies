#%%
# #Загрузка библиотек
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')
# %%
# Пути к файлам
excel_path = r'C:\Users\Олеся\Desktop\kanclerisms_labeled (2).xlsx'
html_path = r'C:\Users\Олеся\Desktop\first model\template.html'

# Загружаем Excel с канцеляризмами и метками
df = pd.read_excel(excel_path)

# %%
#Создаем х и y
X = df['phrase'].astype(str)
y = df['label'].astype(str)

# Делим на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1616)

# Преобразуем текст в числовые векторы TF-IDF
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
# %%
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np

# Диапазон значений регуляризации C
# чем больше C, тем слабее регуляризация
C_values = np.logspace(-3, 3, 15)  # от 0.001 до 1000 (15 точек)

# --- Результаты ---
logreg_results = {
    'C': C_values,
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}

#Обучение модели на разных C
for C in C_values:
    clf = LogisticRegression(C=C, max_iter=500, solver='liblinear', random_state=1630)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    logreg_results['accuracy'].append(accuracy_score(y_test, y_pred))
    logreg_results['precision'].append(precision_score(y_test, y_pred, average='macro'))
    logreg_results['recall'].append(recall_score(y_test, y_pred, average='macro'))
    logreg_results['f1'].append(f1_score(y_test, y_pred, average='macro'))

best_C = 1.5  # из предыдущего анализа
logreg = LogisticRegression(C=best_C, max_iter=500, solver='liblinear', random_state=1630)
logreg.fit(X_train_vec, y_train)
y_pred_logreg = logreg.predict(X_test_vec)

accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg, average='macro', zero_division=0)
recall_logreg = recall_score(y_test, y_pred_logreg, average='macro', zero_division=0)
f1_logreg = f1_score(y_test, y_pred_logreg, average='macro', zero_division=0)

print(f"Accuracy: {accuracy_logreg:.4f}")
print(f"Precision: {precision_logreg:.4f}")
print(f"Recall: {recall_logreg:.4f}")
print(f"F1-score: {f1_logreg:.4f}")
print()
# %%
from sklearn.ensemble import RandomForestClassifier

# Подбор гиперпараметров для Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestClassifier(random_state=1630)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1_macro', n_jobs=-1)
rf_grid.fit(X_train_vec, y_train)

print(f"Лучшие параметры Random Forest: {rf_grid.best_params_}")

best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test_vec)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='macro', zero_division=0)
recall_rf = recall_score(y_test, y_pred_rf, average='macro', zero_division=0)
f1_rf = f1_score(y_test, y_pred_rf, average='macro', zero_division=0)

print(f"Accuracy: {accuracy_rf:.4f}")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-score: {f1_rf:.4f}")
print()

# %%
print("=== Support Vector Machine ===")

# Масштабирование данных для SVM (рекомендуется)
scaler = StandardScaler(with_mean=False)  # with_mean=False для разреженных матриц
X_train_scaled = scaler.fit_transform(X_train_vec)
X_test_scaled = scaler.transform(X_test_vec)

# Упрощенный подбор гиперпараметров для SVM
svm_params = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbf']
}

svm = SVC(random_state=1630)
svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
svm_grid.fit(X_train_scaled, y_train)

print(f"Лучшие параметры SVM: {svm_grid.best_params_}")

best_svm = svm_grid.best_estimator_
y_pred_svm = best_svm.predict(X_test_scaled)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro', zero_division=0)
recall_svm = recall_score(y_test, y_pred_svm, average='macro', zero_division=0)
f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)

print(f"Accuracy: {accuracy_svm:.4f}")
print(f"Precision: {precision_svm:.4f}")
print(f"Recall: {recall_svm:.4f}")
print(f"F1-score: {f1_svm:.4f}")
print()

#%%
#%pip install matplotlib

#%%
import matplotlib.pyplot as plt

# %%
print("=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
models_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
    'Accuracy': [accuracy_logreg, accuracy_rf, accuracy_svm],
    'Precision': [precision_logreg, precision_rf, precision_svm],
    'Recall': [recall_logreg, recall_rf, recall_svm],
    'F1-score': [f1_logreg, f1_rf, f1_svm]
})

print(models_comparison.round(4))

# Визуализация сравнения моделей
plt.figure(figsize=(12, 8))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
x = np.arange(len(metrics))
width = 0.25

plt.bar(x - width, models_comparison.iloc[0, 1:], width, label='Logistic Regression', alpha=0.8)
plt.bar(x, models_comparison.iloc[1, 1:], width, label='Random Forest', alpha=0.8)
plt.bar(x + width, models_comparison.iloc[2, 1:], width, label='SVM', alpha=0.8)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Comparison of Model Performance')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Выбор лучшей модели по F1-score
best_model_idx = models_comparison['F1-score'].idxmax()
best_model_name = models_comparison.loc[best_model_idx, 'Model']
best_model_score = models_comparison.loc[best_model_idx, 'F1-score']

print(f"\n🎯 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с F1-score = {best_model_score:.4f}")
# %%
print("=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
models_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'SVM'],
    'Accuracy': [accuracy_logreg, accuracy_rf, accuracy_svm],
    'Precision': [precision_logreg, precision_rf, precision_svm],
    'Recall': [recall_logreg, recall_rf, recall_svm],
    'F1-score': [f1_logreg, f1_rf, f1_svm]
})

print(models_comparison.round(4))

# Визуализация сравнения моделей - УЛУЧШЕННАЯ ВЕРСИЯ
plt.figure(figsize=(14, 9))

# Синяя цветовая палитра
colors = ['#1f77b4', '#3498db', '#2980b9']  # Оттенки синего
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
x = np.arange(len(metrics))
width = 0.25

# Создание столбцов с улучшенным дизайном
bars1 = plt.bar(x - width, models_comparison.iloc[0, 1:], width,
                label='Logistic Regression',
                color=colors[0],
                edgecolor='white',
                linewidth=1.5,
                alpha=0.9)

bars2 = plt.bar(x, models_comparison.iloc[1, 1:], width,
                label='Random Forest',
                color=colors[1],
                edgecolor='white',
                linewidth=1.5,
                alpha=0.9)

bars3 = plt.bar(x + width, models_comparison.iloc[2, 1:], width,
                label='SVM',
                color=colors[2],
                edgecolor='white',
                linewidth=1.5,
                alpha=0.9)

# Добавление значений на столбцы
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#2c3e50')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# Настройка внешнего вида
plt.xlabel('Метрики', fontsize=12, fontweight='bold', color='#2c3e50')
plt.ylabel('Значение', fontsize=12, fontweight='bold', color='#2c3e50')
plt.title('Сравнение производительности моделей классификации',
          fontsize=14, fontweight='bold', pad=20, color='#2c3e50')
plt.xticks(x, metrics, fontsize=11, fontweight='bold')
plt.yticks(fontsize=10)

# Легенда с улучшенным дизайном
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=3, frameon=True, fancybox=True,
                   shadow=True, fontsize=11)
legend.get_frame().set_facecolor('#ecf0f1')
legend.get_frame().set_alpha(0.9)

# Сетка
plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
plt.ylim(0, 1.1)

# Фон и границы
plt.gca().set_facecolor('#f8f9fa')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('#bdc3c7')
plt.gca().spines['bottom'].set_color('#bdc3c7')

# Добавление горизонтальной линии на уровне 0.5 для ориентира
plt.axhline(y=0.5, color='#e74c3c', linestyle=':', alpha=0.7, linewidth=1)

plt.tight_layout()
plt.show()

# Дополнительная визуализация - тепловая карта метрик
plt.figure(figsize=(10, 6))
metrics_data = models_comparison.set_index('Model').T

# Создание тепловой карты
plt.imshow(metrics_data, cmap='Blues', aspect='auto', vmin=0, vmax=1)

# Настройка осей
plt.xticks(range(len(metrics_data.columns)), metrics_data.columns, rotation=45, ha='right')
plt.yticks(range(len(metrics_data.index)), metrics_data.index)

# Добавление значений в ячейки
for i in range(len(metrics_data.index)):
    for j in range(len(metrics_data.columns)):
        plt.text(j, i, f'{metrics_data.iloc[i, j]:.3f}',
                ha='center', va='center', fontweight='bold',
                color='white' if metrics_data.iloc[i, j] > 0.6 else 'black')

plt.colorbar(label='Значение метрики')
plt.title('Тепловая карта метрик моделей', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Модели', fontweight='bold')
plt.ylabel('Метрики', fontweight='bold')
plt.tight_layout()
plt.show()

# Выбор лучшей модели по F1-score
best_model_idx = models_comparison['F1-score'].idxmax()
best_model_name = models_comparison.loc[best_model_idx, 'Model']
best_model_score = models_comparison.loc[best_model_idx, 'F1-score']

print(f"\n🎯 ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с F1-score = {best_model_score:.4f}")

# Дополнительная информация о лучшей модели
best_model_row = models_comparison.loc[best_model_idx]
print(f"📊 Полные метрики лучшей модели:")
print(f"   • Accuracy: {best_model_row['Accuracy']:.4f}")
print(f"   • Precision: {best_model_row['Precision']:.4f}")
print(f"   • Recall: {best_model_row['Recall']:.4f}")
print(f"   • F1-score: {best_model_row['F1-score']:.4f}")
# %%
