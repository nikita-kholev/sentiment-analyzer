import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import uuid
import tempfile
import logging
from datetime import datetime

try:
    from model import SentimentAnalyzer
    ML_MODEL_LOADED = True
except ImportError as e:
    logging.warning(f"ML model not loaded: {e}")
    from simple_model import RuleBasedSentimentAnalyzer as SentimentAnalyzer
    ML_MODEL_LOADED = False

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Инициализация анализатора
analyzer = SentimentAnalyzer()

# Временное хранилище для результатов
results_store = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка статуса API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_type': 'ml' if ML_MODEL_LOADED else 'rule_based',
        'python_version': sys.version,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze/csv', methods=['POST'])
def analyze_csv():
    """Анализ CSV файла"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Файл не найден'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Файл должен быть в формате CSV'}), 400
        
        # Чтение CSV файла
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'error': f'Ошибка чтения CSV: {str(e)}'}), 400
        
        if df.empty:
            return jsonify({'error': 'CSV файл пуст'}), 400
        
        # Анализ с помощью модели
        analysis_result = analyzer.analyze_csv(df)
        
        if not analysis_result['success']:
            return jsonify({'error': analysis_result['error']}), 500
        
        # Сохранение результатов
        download_token = str(uuid.uuid4())
        
        # Сохраняем результат в CSV
        temp_dir = tempfile.gettempdir()
        output_filename = f"sentiment_analysis_{download_token}.csv"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Сохраняем в формате согласно ТЗ
        output_df = analysis_result['results_df']
        if 'ID' in output_df.columns:
            output_df[['ID', 'score']].to_csv(output_path, index=False, encoding='utf-8')
        else:
            first_col = output_df.columns[0]
            output_df[[first_col, 'score']].to_csv(output_path, index=False, encoding='utf-8')
        
        # Сохраняем полную версию
        full_output_path = os.path.join(temp_dir, f"full_{output_filename}")
        analysis_result['results_df'].to_csv(full_output_path, index=False, encoding='utf-8')
        
        # Сохраняем информацию о файле
        results_store[download_token] = {
            'filename': output_filename,
            'filepath': output_path,
            'full_filepath': full_output_path,
            'created_at': datetime.now(),
            'statistics': analysis_result['statistics']
        }
        
        # Очистка старых файлов
        cleanup_old_files()
        
        return jsonify({
            'success': True,
            'download_token': download_token,
            'statistics': analysis_result['statistics'],
            'preview_data': analysis_result['preview_data'],
            'message': f'Проанализировано {len(df)} отзывов',
            'model_type': 'ml' if ML_MODEL_LOADED else 'rule_based'
        })
        
    except Exception as e:
        logger.error(f"Ошибка при анализе CSV: {str(e)}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/api/download/<download_token>', methods=['GET'])
def download_results(download_token):
    """Скачивание результатов анализа"""
    try:
        if download_token not in results_store:
            return jsonify({'error': 'Файл не найден или устарел'}), 404
        
        file_info = results_store[download_token]
        filepath = file_info['filepath']
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'Файл не найден'}), 404
        
        filename = f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Ошибка при скачивании: {str(e)}")
        return jsonify({'error': 'Ошибка при скачивании файла'}), 500

def cleanup_old_files():
    """Очистка старых файлов результатов"""
    try:
        current_time = datetime.now()
        tokens_to_remove = []
        
        for token, info in results_store.items():
            if (current_time - info['created_at']).total_seconds() > 3600:
                tokens_to_remove.append(token)
                for filepath in [info['filepath'], info.get('full_filepath')]:
                    if filepath and os.path.exists(filepath):
                        os.remove(filepath)
        
        for token in tokens_to_remove:
            del results_store[token]
            
    except Exception as e:
        logger.error(f"Ошибка при очистке файлов: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)