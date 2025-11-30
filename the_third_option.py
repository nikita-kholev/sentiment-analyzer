import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import time

def preprocess_text(text):
    """Предобработка текста с учетом русского языка"""
    if isinstance(text, float) or isinstance(text, int):
        text = str(text)
    
    # Базовая очистка
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def train_model(train_file, model_file):
    # Чтение с правильной кодировкой
    train_df = pd.read_csv(train_file, encoding='utf-8-sig')
    train_df['text'] = train_df['text'].fillna('')
    
    # Применяем предобработку
    train_df['text'] = train_df['text'].apply(preprocess_text)
    
    # Быстрые алгоритмы для многоклассовой классификации
    models = {
        'LogisticRegression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            n_jobs=-1
        ),
        'MultinomialNB': MultinomialNB(),
        'LinearSVC': LinearSVC(
            random_state=42, 
            max_iter=1000
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=50,  # Меньше деревьев для скорости
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_score = 0
    best_model = None
    best_name = None
    
    print("Сравнение моделей:")
    
    for name, model in models.items():
        start_time = time.time()
        
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2
            )),
            ('classifier', model)
        ])
        
        # Быстрая кросс-валидация
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(
            pipeline, 
            train_df['text'], 
            train_df['label'], 
            cv=cv, 
            scoring='accuracy',
            n_jobs=1  # Для стабильности
        )
        
        mean_score = scores.mean()
        training_time = time.time() - start_time
        
        print(f"{name}: accuracy={mean_score:.4f}, time={training_time:.2f}s")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model = pipeline
            best_name = name
    
    print(f"\nЛучшая модель: {best_name} с accuracy {best_score:.4f}")
    
    # Обучаем лучшую модель на всех данных
    best_model.fit(train_df['text'], train_df['label'])
    
    # Сохраняем модель
    joblib.dump(best_model, model_file)
    
    return best_model

def predict_model(model_file, test_file):
    model = joblib.load(model_file)
    test_df = pd.read_csv(test_file, encoding='utf-8-sig')
    test_df['text'] = test_df['text'].fillna('')
    test_df['text'] = test_df['text'].apply(preprocess_text)
    
    predictions = model.predict(test_df['text'])
    return predictions.tolist()