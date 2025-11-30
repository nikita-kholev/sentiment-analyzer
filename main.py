from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from scipy import sparse
import re
import io
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CSV Sentiment Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    return FileResponse('static/index.html')

class AdvancedSentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.best_model_name = ""
        
        self.positive_words = {
            'отлично', 'прекрасно', 'супер', 'рекомендую', 'восхитительно', 'потрясающе',
            'замечательно', 'хорошо', 'удобно', 'качественно', 'быстро', 'отличный',
            'прекрасный', 'великолепно', 'доволен', 'рад', 'нравится', 'люблю', 'шикарно',
            'бесподобно', 'идеально', 'превосходно', 'чудесно', 'замечательный', 'хороший',
            'удобный', 'качественный', 'быстрый', 'надежный', 'профессионально', 'восхищение',
            'восторг', 'блестяще', 'безупречно', 'совершенно', 'прекрасно', 'отлично'
        }
        
        self.negative_words = {
            'ужасно', 'кошмар', 'отвратительно', 'разочарован', 'не советую', 'плохо',
            'медленно', 'сложно', 'дорого', 'брак', 'некачественно', 'ужасный',
            'отвратный', 'мерзко', 'неудобно', 'раздражает', 'бесит', 'не нравится',
            'упаднически', 'неприемлемо', 'хреново', 'гадко', 'паршиво', 'никуда',
            'сломалось', 'не работает', 'глючит', 'тормозит', 'переплатил', 'разочарование',
            'недостаток', 'проблема', 'ошибка', 'баг', 'дефект', 'неисправность'
        }
        
        self.noise_indicators = {
            '...', '???', '!!!', '??', '!!', 'как', 'что', 'где', 'когда', 'почему',
            'зачем', 'сколько', 'кто', 'чем', 'надо', 'нужно', 'можно', 'нельзя',
            'возможно', 'вероятно', 'кажется', 'наверное', 'пожалуйста', 'спасибо',
            'привет', 'здравствуйте', 'до свидания', 'пока', 'ок', 'ладно', 'хорошо',
            'понятно', 'ясно', 'да', 'нет', 'не знаю', 'не уверен'
        }

    def detect_noise(self, text: str) -> bool:
        text_lower = text.lower().strip()
        
        if len(text_lower) < 10:
            return True
            
        words = set(re.findall(r'\b\w+\b', text_lower))
        if len(words.intersection(self.noise_indicators)) > 2:
            return True
            
        if re.search(r'(.)\1{3,}', text_lower):
            return True
            
        if len(re.findall(r'\b[а-я]+\b', text_lower)) < 3:
            return True
            
        return False

    def create_advanced_features(self, texts: List[str]) -> np.ndarray:
        features = []
        
        for text in texts:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            pos_count = sum(1 for word in words if word in self.positive_words)
            neg_count = sum(1 for word in words if word in self.negative_words)
            
            exclamation_count = text.count('!')
            question_count = text.count('?')
            capital_count = sum(1 for c in text if c.isupper())
            
            total_words = max(len(words), 1)
            unique_words = len(set(words))
            lexical_diversity = unique_words / total_words
            
            is_noise = self.detect_noise(text)
            text_length = len(text)
            
            pos_ratio = pos_count / total_words
            neg_ratio = neg_count / total_words
            sentiment_balance = (pos_count - neg_count) / total_words
            emotion_intensity = (pos_count + neg_count) / total_words
            
            features.append([
                pos_count, neg_count, 
                pos_ratio, neg_ratio, 
                sentiment_balance, emotion_intensity,
                exclamation_count, question_count, capital_count,
                lexical_diversity, text_length, total_words,
                int(is_noise)
            ])
        
        return np.array(features)

    def train_advanced_model(self, texts: List[str], labels: List[int]) -> Dict[str, Any]:
        logger.info("Запуск оптимизированного обучения с 4 лучшими алгоритмами...")
        
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.85,
                stop_words=None,
                analyzer='word'
            )
            
            X_tfidf = self.vectorizer.fit_transform(texts)
            X_advanced = self.create_advanced_features(texts)
            X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_advanced)])
            
            # Оставляем только 4 лучшие модели на основе предыдущих результатов
            models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=25,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=10,
                    learning_rate=0.1,
                    random_state=42
                ),
                'logistic_l2': LogisticRegression(
                    C=0.8,
                    solver='liblinear',
                    max_iter=2000,
                    random_state=42,
                    class_weight='balanced'
                ),
                'naive_bayes': MultinomialNB(alpha=0.1)
            }
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            best_score = 0
            best_model = None
            best_model_name = ""
            all_results = {}
            
            for name, model in models.items():
                logger.info(f"Тестирование {name}...")
                start_time = time.time()
                
                try:
                    cv_scores = cross_val_score(
                        model, X_combined, labels, 
                        cv=skf, scoring='f1_macro', n_jobs=-1
                    )
                    
                    model.fit(X_combined, labels)
                    
                    if not hasattr(model, 'predict_proba'):
                        calibrated_model = CalibratedClassifierCV(model, cv=3)
                        calibrated_model.fit(X_combined, labels)
                        final_model = calibrated_model
                    else:
                        final_model = model
                    
                    training_time = time.time() - start_time
                    
                    all_results[name] = {
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'cv_scores': cv_scores.tolist(),
                        'training_time': training_time,
                        'model': final_model
                    }
                    
                    logger.info(f"{name}: F1-macro = {cv_scores.mean():.4f} ± {cv_scores.std():.4f} (время: {training_time:.2f}с)")
                    
                    if cv_scores.mean() > best_score:
                        best_score = cv_scores.mean()
                        best_model = final_model
                        best_model_name = name
                        
                except Exception as e:
                    logger.error(f"Ошибка при обучении {name}: {str(e)}")
                    continue
            
            self.model = best_model
            self.best_model_name = best_model_name
            self.is_trained = True
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_combined, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            detailed_f1 = f1_score(y_test, y_pred, average='macro')
            
            logger.info(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name} с F1-macro = {best_score:.4f}")
            logger.info(f"Тестовый F1-macro: {detailed_f1:.4f}")
            
            return {
                'success': True,
                'best_model': best_model_name,
                'best_score': best_score,
                'test_score': detailed_f1,
                'all_results': all_results,
                'feature_shape': X_combined.shape
            }
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {str(e)}")
            return {'success': False, 'error': str(e)}

    def predict_with_noise_handling(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.is_trained:
            raise ValueError("Модель не обучена")
        
        X_tfidf = self.vectorizer.transform(texts)
        X_advanced = self.create_advanced_features(texts)
        X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_advanced)])
        
        predictions = self.model.predict(X_combined)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_combined)
            confidence = np.max(probabilities, axis=1)
            noise_threshold = 0.6
        else:
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X_combined)
                if scores.ndim == 1:
                    confidence = 1 / (1 + np.exp(-np.abs(scores)))
                else:
                    confidence = np.max(self._softmax(scores), axis=1)
                noise_threshold = 0.55
            else:
                confidence = np.ones(len(texts)) * 0.8
                noise_threshold = 0.6
        
        results = []
        for i, text in enumerate(texts):
            original_sentiment = int(predictions[i])
            conf = float(np.clip(confidence[i], 0.1, 0.99))
            
            is_noisy = self.detect_noise(text)
            if conf < noise_threshold or is_noisy:
                final_sentiment = 0
                was_adjusted = True
            else:
                final_sentiment = original_sentiment
                was_adjusted = False
            
            sentiment_label = ['neutral', 'positive', 'negative'][final_sentiment]
            sentiment_display = ['Нейтральный', 'Позитивный', 'Негативный'][final_sentiment]
            
            results.append({
                'text': text,
                'sentiment': final_sentiment,
                'sentiment_label': sentiment_label,
                'sentiment_display': sentiment_display,
                'confidence': conf,
                'original_sentiment': original_sentiment,
                'was_adjusted': was_adjusted,
                'is_noise': is_noisy
            })
        
        return results

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

analyzer = AdvancedSentimentAnalyzer()

def create_enhanced_sample_data():
    sample_texts = []
    sample_labels = []
    
    neutral_samples = [
        "Нормальный товар обычного качества", "Стандартный продукт ничего особенного",
        "Приемлемо за такие деньги", "Обычно как и ожидал", "Неплохо но есть лучшие варианты",
        "Средненько нормально в целом", "Ничего выдающегося но и не плохо",
        "Удовлетворительно соответствует цене", "Нейтрально без сильных эмоций",
        "Обычный товар как все", "Нормально работает в принципе",
        "спасибо", "хорошо", "не знаю", "нормально", "ок", "ладно",
        "???", "!!!", "....", "да нет наверное", "возможно", "кажется да",
        "привет как дела", "до свидания", "пока", "ясно понятно",
        "норм", "так себе", "ничего", "без комментариев", "нет слов"
    ]
    
    positive_samples = [
        "Отличный товар очень доволен покупкой", "Качественно сделано рекомендую всем", 
        "Быстрая доставка хорошее обслуживание", "Прекрасный продукт работает идеально",
        "Очень удобно и практично спасибо огромное", "Супер качество превзошло все ожидания",
        "Великолепно буду рекомендовать друзьям и знакомым", "Быстро качественно профессионально",
        "Отличное соотношение цены и качества просто супер", "Потрясающий результат очень рад покупке",
        "Замечательный сервис всем доволен полностью", "Идеально подошло спасибо большое",
        "Работает без нареканий отлично рекомендую", "Очень хороший продукт советую всем",
        "Качество на высшем уровне просто восхитительно", "Быстро привезли упаковано хорошо",
        "Полностью соответствует описанию очень доволен", "Очень порадовало качество просто супер",
        "Просто супер всем рекомендую не пожалеете", "Лучшее что покупал просто великолепно"
    ]
    
    negative_samples = [
        "Ужасное качество никогда не советую такое", "Очень разочарован деньги на ветер просто кошмар",
        "Товар сломался через день полный кошмар", "Плохое обслуживание никогда не рекомендую",
        "Дорого и некачественно просто мерзко", "Никуда не годится полный провал разочарование",
        "Ужасный сервис никогда больше не обращусь", "Не работает как должно бесит и раздражает",
        "Очень медленно тормозит постоянно глючит", "Отвратительное качество брак полный",
        "Полный разочарование не покупайте никогда", "Обман не соответствует описанию вообще",
        "Неработающий товар верните деньги сейчас", "Ужасная доставка ждал неделю просто кошмар",
        "Качество ниже плинтуса просто ужасно", "Не стоит таких денег вообще никогда",
        "Постоянные глюки не работает отвратительно", "Разочаровала покупка очень сильно",
        "Худший сервис что видел просто ужас", "Не рекомендую это гадость полная"
    ]
    
    sample_texts.extend(neutral_samples)
    sample_labels.extend([0] * len(neutral_samples))
    
    sample_texts.extend(positive_samples)
    sample_labels.extend([1] * len(positive_samples))
    
    sample_texts.extend(negative_samples)
    sample_labels.extend([2] * len(negative_samples))
    
    return sample_texts, sample_labels

@app.on_event("startup")
async def startup_event():
    logger.info("Загрузка и обучение оптимизированной модели...")
    
    sample_texts, sample_labels = create_enhanced_sample_data()
    training_result = analyzer.train_advanced_model(sample_texts, sample_labels)
    
    if training_result['success']:
        logger.info(f"Модель успешно обучена. Лучшая модель: {training_result['best_model']}")
        logger.info(f"CV F1-macro: {training_result['best_score']:.4f}")
        logger.info(f"Test F1-macro: {training_result['test_score']:.4f}")
        
        for name, result in training_result['all_results'].items():
            logger.info(f"  {name}: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    else:
        logger.error(f"Ошибка при обучении модели: {training_result['error']}")

@app.post("/api/analyze")
async def analyze_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Формат файла должен быть CSV")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        text_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                       ['text', 'review', 'comment', 'отзыв', 'комментарий', 'текст'])]
        
        if not text_columns:
            raise HTTPException(status_code=400, detail="Не найдена колонка с текстом отзывов")
        
        text_column = text_columns[0]
        texts = df[text_column].fillna('').astype(str).tolist()
        
        if not texts:
            raise HTTPException(status_code=400, detail="Файл не содержит текстовых данных")
        
        logger.info(f"Обработка {len(texts)} отзывов с улучшенной моделью {analyzer.best_model_name}...")
        
        results = analyzer.predict_with_noise_handling(texts)
        
        sentiment_counts = {'neutral': 0, 'positive': 0, 'negative': 0}
        adjusted_counts = 0
        noise_counts = 0
        confidence_scores = []
        
        for result in results:
            sentiment_counts[result['sentiment_label']] += 1
            if result['was_adjusted']:
                adjusted_counts += 1
            if result['is_noise']:
                noise_counts += 1
            confidence_scores.append(result['confidence'])
        
        confidence_stats = {
            'average': np.mean(confidence_scores) if confidence_scores else 0,
            'max': np.max(confidence_scores) if confidence_scores else 0,
            'min': np.min(confidence_scores) if confidence_scores else 0,
            'std': np.std(confidence_scores) if confidence_scores else 0
        }
        
        id_column = None
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'index', 'номер']):
                id_column = col
                break
        
        if id_column:
            result_df = pd.DataFrame({
                'id': df[id_column],
                'label': [r['sentiment'] for r in results]
            })
        else:
            result_df = pd.DataFrame({
                'id': range(1, len(results) + 1),
                'label': [r['sentiment'] for r in results]
            })
        
        csv_output = result_df.to_csv(index=False)
        
        preview = results[:10]
        
        return JSONResponse({
            'success': True,
            'model_used': analyzer.best_model_name,
            'statistics': {
                'total_samples': len(texts),
                'sentiment_distribution': sentiment_counts,
                'adjusted_samples': adjusted_counts,
                'noise_samples': noise_counts,
                'confidence': confidence_stats
            },
            'preview': preview,
            'results_csv': csv_output
        })
        
    except Exception as e:
        logger.error(f"Ошибка при анализе файла: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": analyzer.is_trained,
        "best_model": analyzer.best_model_name,
        "service": "Enhanced CSV Sentiment Analyzer"
    }

@app.get("/api/model_info")
async def model_info():
    if not analyzer.is_trained:
        raise HTTPException(status_code=400, detail="Модель не обучена")
    
    return {
        "best_model": analyzer.best_model_name,
        "is_trained": analyzer.is_trained
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)