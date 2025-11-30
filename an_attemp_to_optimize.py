# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import re
import io
import uuid
import time
import os
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizedSentimentAnalyzer:
    def __init__(self):
        # Оптимизированные параметры
        self.vectorizer = TfidfVectorizer(
            max_features=3000,  # Уменьшил для скорости
            ngram_range=(1, 2),
            min_df=5,  # Более строгая фильтрация редких слов
            max_df=0.9,
            lowercase=True,
            analyzer='word'
        )
        self.model = None
        self.is_trained = False
        self.best_f1 = 0
        
        # ОПТИМИЗИРОВАННЫЙ набор стоп-слов и фильтров
        self.stop_words = {
            # Базовые стоп-слова
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его',
            'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
            'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже',
            
            # Электронная коммерция
            'товар', 'продукт', 'покупка', 'заказ', 'доставка', 'продавец', 'магазин', 'цена', 'рубль', 'руб',
            'шт', 'штука', 'размер', 'цвет', 'качество', 'сервис', 'упаковка', 'курьер', 'отправка', 'получение',
            
            # Сленг и частые слова
            'привет', 'пока', 'спасибо', 'пожалуйста', 'извините', 'ок', 'окей', 'ладно', 'хорошо', 'понятно',
            'короче', 'типа', 'как бы', 'значит', 'вот', 'так сказать', 'фигня', 'хрень', 'ерунда', 'бред',
            'штука', 'фишка', 'прикол', 'норм', 'офигенно', 'отстой', 'лажа', 'круто', 'супер', 'ужас', 'кошмар',
        }
        
        # Компилируем regex для скорости
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.email_pattern = re.compile(r'\S*@\S*\s?')
        self.phone_pattern = re.compile(r'[\+\(\)\-\d\s]{10,}')
        self.non_russian_pattern = re.compile(r'[^а-яё\s]')
        self.space_pattern = re.compile(r'\s+')
        self.digit_in_word_pattern = re.compile(r'\d+')  # Для поиска цифр в словах
    
    def fast_preprocess(self, text):
        """СУПЕР БЫСТРАЯ предобработка с оптимизацией"""
        if pd.isna(text) or not text or text == '':
            return ""
        
        # Быстрое преобразование в строку и очистка
        text = str(text).lower().strip()
        
        if not text:  # После strip может стать пустым
            return ""
        
        # ОДНОВРЕМЕННАЯ очистка (быстрее отдельных замен)
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        text = self.phone_pattern.sub('', text)
        text = self.non_russian_pattern.sub(' ', text)
        text = self.space_pattern.sub(' ', text).strip()
        
        if not text:
            return ""
        
        # ОПТИМИЗИРОВАННАЯ фильтрация слов
        words = []
        for word in text.split():
            word_len = len(word)
            
            # БЫСТРАЯ проверка условий (самые частые случаи сначала)
            if word_len < 3 or word_len > 25:
                continue
                
            if word in self.stop_words:
                continue
                
            if word.isdigit():
                continue
                
            # Проверка цифр в слове (только если нужно)
            if self.digit_in_word_pattern.search(word):
                continue
                
            words.append(word)
        
        return ' '.join(words)
    
    def train_model(self, texts: List[str], labels: List[int]):
        """Оптимизированное обучение"""
        start_time = time.time()
        print("Быстрое обучение модели...")
        
        # Быстрая предобработка
        processed_texts = []
        for i, text in enumerate(texts):
            processed = self.fast_preprocess(text)
            if processed:  # Только непустые тексты
                processed_texts.append(processed)
        
        # Если после предобработки остались пустые тексты, используем оригиналы
        if len(processed_texts) < len(texts):
            print(f"После предобработки осталось {len(processed_texts)}/{len(texts)} текстов")
            # Используем оригинальные тексты для оставшихся
            for i, text in enumerate(texts):
                if not processed_texts or i >= len(processed_texts):
                    processed_texts.append(self.fast_preprocess(text) or " ")
        
        print(f"Обработано {len(processed_texts)} текстов")
        
        # Векторизация
        X = self.vectorizer.fit_transform(processed_texts)
        y = np.array(labels)
        
        print(f"📊 Создано {X.shape[1]} признаков")
        
        # Быстрая кросс-валидация
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Только лучшая модель
        model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=500,  # Меньше итераций
            random_state=42,
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        # Кросс-валидация
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
        mean_f1 = np.mean(scores)
        
        print(f"📊 Cross-val Macro-F1: {mean_f1:.4f} (+/- {np.std(scores):.4f})")
        
        # Обучение
        self.model = model
        self.model.fit(X, y)
        self.is_trained = True
        self.best_f1 = mean_f1
        
        # Быстрая оценка
        y_pred = self.model.predict(X)
        final_f1 = f1_score(y, y_pred, average='macro')
        
        training_time = time.time() - start_time
        
        print(f"Обучение завершено за {training_time:.2f} сек")
        print(f"Final Macro-F1: {final_f1:.4f}")
        
        return {
            "best_model": "logistic_regression",
            "cross_val_f1": float(mean_f1),
            "final_f1": float(final_f1),
            "training_time": training_time
        }
    
    def predict(self, texts: List[str]):
        """Сверхбыстрое предсказание"""
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        # Быстрая предобработка
        processed_texts = [self.fast_preprocess(text) or " " for text in texts]
        
        X = self.vectorizer.transform(processed_texts)
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores, probabilities

# Глобальная модель
analyzer = OptimizedSentimentAnalyzer()

def load_and_train_model():
    """Автоматическая загрузка и обучение"""
    try:
        if not os.path.exists('train.csv'):
            print("Файл train.csv не найден!")
            return False
        
        print("📁 Загружаем train.csv...")
        df = pd.read_csv('train.csv')
        
        # Проверка колонок
        required_cols = ['ID', 'text', 'src', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Отсутствуют колонки: {missing_cols}")
            return False
        
        # Подготовка данных (игнорируем src, как просили)
        texts = df['text'].fillna('').astype(str).tolist()
        labels = df['label'].astype(int).tolist()
        
        # Быстрая проверка меток
        valid_labels = {0, 1, 2}
        if not all(label in valid_labels for label in labels):
            print("❌ Найдены недопустимые метки")
            return False
        
        print(f"📦 Загружено {len(texts)} samples")
        
        # Обучение
        metrics = analyzer.train_model(texts, labels)
        
        print("🎉 Модель готова к работе!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

@app.post("/api/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    """Анализ CSV файлов"""
    try:
        if not analyzer.is_trained:
            raise HTTPException(400, "Модель не обучена")
        
        start_time = time.time()
        contents = await file.read()
        
        # Быстрое чтение
        df = None
        for encoding in ['utf-8', 'cp1251', 'windows-1251']:
            try:
                df = pd.read_csv(io.StringIO(contents.decode(encoding)))
                break
            except:
                continue
        
        if df is None:
            raise HTTPException(400, "Не удалось декодировать файл")
        
        # Поиск текстовой колонки
        text_column = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['text', 'текст', 'review', 'отзыв']):
                text_column = col
                break
        
        if not text_column:
            text_column = df.columns[0]
        
        # Добавляем ID если нужно
        if 'ID' not in df.columns:
            df['ID'] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
        
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # Быстрое предсказание
        predictions, confidence_scores, probabilities = analyzer.predict(texts)
        
        # Результаты
        results = []
        for i, (text, pred, conf) in enumerate(zip(texts, predictions, confidence_scores)):
            results.append({
                'ID': str(df['ID'].iloc[i]),
                'text': text[:80] + '...' if len(text) > 80 else text,
                'sentiment': int(pred),
                'confidence': float(conf)
            })
        
        # Статистика для диаграммы
        sentiment_counts = {
            'neutral': int((predictions == 0).sum()),
            'positive': int((predictions == 1).sum()),
            'negative': int((predictions == 2).sum())
        }
        
        total = len(predictions)
        sentiment_percentages = {
            'neutral': round(sentiment_counts['neutral'] / total * 100, 1) if total > 0 else 0,
            'positive': round(sentiment_counts['positive'] / total * 100, 1) if total > 0 else 0,
            'negative': round(sentiment_counts['negative'] / total * 100, 1) if total > 0 else 0
        }
        
        # Итоговый CSV
        result_df = pd.DataFrame([{'ID': r['ID'], 'sentiment': r['sentiment']} for r in results])
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        csv_content = output.getvalue()
        
        processing_time = time.time() - start_time
        
        return JSONResponse({
            "success": True,
            "processing_time_seconds": round(processing_time, 2),
            "statistics": {
                "total_samples": len(df),
                "sentiment_distribution": sentiment_counts,
                "sentiment_percentages": sentiment_percentages,
                "confidence_avg": float(np.mean(confidence_scores))
            },
            "preview": results[:8],
            "results_csv": csv_content
        })
        
    except Exception as e:
        raise HTTPException(500, f"Ошибка: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_trained": analyzer.is_trained,
        "macro_f1_score": analyzer.best_f1 if analyzer.is_trained else None
    }

@app.on_event("startup")
async def startup_event():
    print("🚀 Запуск API...")
    if load_and_train_model():
        print("Сервер готов!")
    else:
        print("Ошибка обучения")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)