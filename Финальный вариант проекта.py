#%%
# 2. Загружаем библиотеки и ресурсы
import pandas as pd
import nltk
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from spellchecker import SpellChecker

# Скачиваем токенизаторы NLTK
nltk.download('punkt')
nltk.download('punkt_tab')

# Инициализируем проверку орфографии
spell = SpellChecker(language='ru')

#%%
# Пути к файлам
excel_path = r'C:\Users\Олеся\Desktop\kanclerisms_labeled (2).xlsx'
html_path = r'C:\Users\Олеся\Desktop\first model\template.html'

# Загружаем Excel файл с канцеляризмами и метками
df = pd.read_excel(excel_path)
# %%
# 3. Извлекаем текст из HTML
with open(html_path, 'r', encoding='utf-8') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')
text = soup.get_text(separator=' ')
# %%
from sklearn.ensemble import RandomForestClassifier
# 4. Обучаем модель randomforrest регрессии для канцеляризмов
X = df['phrase'].astype(str)
y = df['label'].astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1616)

vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

rf = RandomForestClassifier(random_state=1630, max_depth = 20, min_samples_split = 10, n_estimators = 200)
rf.fit(X_train_vec, y_train)
# %%
#Подсвечиваем канцеляризмы и орфограммы
def highlight_all(html, threshold=0.7):
    soup = BeautifulSoup(html, 'html.parser')
    text_nodes = soup.find_all(string=True)

    for node in text_nodes:
        words = nltk.word_tokenize(node, language='russian')
        new_node = node

        for w in set(words):
            # Пропускаем короткие слова и слова с неалфавитными символами
            if len(w) < 3 or not w.isalpha():
                continue

            # Проверяем канцеляризм через ML модель
            prob = rf.predict_proba(vectorizer.transform([w]))[0]
            idx = list(rf.classes_).index('канцеляризм')
            is_kanclerism = prob[idx] > threshold

            # Проверяем орфографическую ошибку
            is_spelling_error = w.lower() not in spell

            # Выбираем цвет
            if is_kanclerism and is_spelling_error:
                color = '#ffb347'  #оранжевый (оба признака)
            elif is_kanclerism:
                color = '#ffec99'  #жёлтый (канцеляризм)
            elif is_spelling_error:
                color = '#ff9999'  #красный (ошибка)
            else:
                continue  # ничего не выделяем

            # Подсвечиваем слово
            new_node = new_node.replace(
                w, f'<mark style="background-color:{color};">{w}</mark>'
            )

        node.replace_with(BeautifulSoup(new_node, 'html.parser'))

    return str(soup)


highlighted_html = highlight_all(html_content)
# %%
output_path = r"C:\Users\Олеся\Desktop\content\result1.html"

# Создаём папку, если её ещё нет
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 💾 Сохраняем результат
with open(output_path, "w", encoding="utf-8") as f:
    f.write(highlighted_html)

# %%
