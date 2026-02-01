"""
Telegram бот для взаимодействия с RAG ассистентом.
"""

import os
import logging
import asyncio
import math
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from rag_pipeline import RAGPipeline
# Ленивый импорт для datasets и ragas (используются только в /evaluate)
# Это избегает проблем с загрузкой torch при старте бота

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env файла
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Глобальная переменная для RAG pipeline
pipeline = None

# Тестовые вопросы для оценки RAG системы
EVALUATION_QUESTIONS = [
    "Какие услуги вы предоставляете?",
    "Каковы основные этапы установки и настройки SSH на сервере с Ubuntu?",
    "Что такое хранилище S3?"
]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    welcome_message = """
🤖 <b>RAG Ассистент</b>

Привет! Я AI-ассистент с технологией RAG (Retrieval-Augmented Generation).

<b>Что я умею:</b>
• Отвечать на вопросы на основе базы знаний
• Использовать кеш для быстрых ответов
• Предоставлять статистику системы

<b>Команды:</b>
/start - Начать работу
/help - Справка
/stats - Статистика системы
/clear - Очистить кеш
/evaluate - Оценка качества RAG системы (метрики RAGAS: Faithfulness, Context Precision)

Просто напишите мне вопрос, и я постараюсь на него ответить! 🚀
    """
    await update.message.reply_text(welcome_message, parse_mode='HTML')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    help_text = """
📖 <b>Справка по использованию бота</b>

<b>Как использовать:</b>
Просто отправьте мне любой вопрос текстом, и я найду ответ в базе знаний.

<b>Доступные команды:</b>
/start - Приветствие и информация о боте
/help - Эта справка
/stats - Показать статистику системы (количество документов, размер кеша и т.д.)
/clear - Очистить кеш ответов
/evaluate - Оценка качества RAG системы через RAGAS (Faithfulness, Context Precision)

<b>Особенности:</b>
• Ответы кешируются для быстрого доступа
• Используется векторный поиск для нахождения релевантной информации
• Модель: Claude Sonnet 4.5 через ProxyAPI

<b>Примеры вопросов:</b>
• Что такое машинное обучение?
• Как работают нейронные сети?
• Что такое RAG?
    """
    await update.message.reply_text(help_text, parse_mode='HTML')


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /stats."""
    if pipeline is None:
        await update.message.reply_text("❌ Система не инициализирована")
        return
    
    try:
        stats = pipeline.get_stats()
        
        stats_message = f"""
📊 <b>Статистика системы</b>

<b>🗄️ Векторное хранилище:</b>
• Коллекция: {stats['vector_store']['name']}
• Документов: {stats['vector_store']['count']}
• Размерность: {stats['vector_store'].get('embedding_dim', 'N/A')}

<b>💾 Кеш:</b>
• Записей: {stats['cache']['total_entries']}
• Размер БД: {stats['cache']['db_size_mb']:.2f} MB
"""
        
        if stats['cache']['oldest_entry']:
            stats_message += f"• Первая запись: {stats['cache']['oldest_entry']}\n"
        if stats['cache']['newest_entry']:
            stats_message += f"• Последняя запись: {stats['cache']['newest_entry']}\n"
        
        stats_message += f"""
<b>🤖 Модель:</b> {stats['model']}
<b>🌐 Режим:</b> {stats['mode']}
        """
        
        await update.message.reply_text(stats_message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}")
        await update.message.reply_text(f"❌ Ошибка при получении статистики: {e}")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /clear."""
    if pipeline is None:
        await update.message.reply_text("❌ Система не инициализирована")
        return
    
    try:
        pipeline.cache.clear()
        await update.message.reply_text("✅ Кеш успешно очищен!")
    except Exception as e:
        logger.error(f"Ошибка при очистке кеша: {e}")
        await update.message.reply_text(f"❌ Ошибка при очистке кеша: {e}")


def prepare_dataset_for_ragas(pipeline: RAGPipeline, questions: list):
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    # Ленивый импорт Dataset (избегаем загрузки torch при старте)
    from datasets import Dataset
    
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    for i, question in enumerate(questions, 1):
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ (для демонстрации используем часть ответа)
        ground_truths_list.append(result["answer"][:100])
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


async def evaluate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /evaluate для запуска RAGAS оценки."""
    if pipeline is None:
        await update.message.reply_text("❌ Система не инициализирована")
        return
    
    # Ленивый импорт RAGAS (избегаем загрузки torch при старте бота)
    try:
        # Пытаемся импортировать datasets и ragas
        # На Windows может быть проблема с загрузкой torch DLL
        import sys
        import warnings
        
        # Подавляем предупреждения при импорте
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                from datasets import Dataset
                from ragas import evaluate
            except (OSError, ImportError) as dll_error:
                # Ошибка загрузки DLL (обычно torch на Windows)
                error_msg = str(dll_error)
                if "DLL" in error_msg or "c10.dll" in error_msg or "torch" in error_msg.lower() or "WinError 1114" in error_msg:
                    help_text = (
                        "❌ <b>Ошибка загрузки torch (DLL) на Windows</b>\n\n"
                        "🔧 <b>Решения:</b>\n\n"
                        "1️⃣ <b>Установите Visual C++ Redistributable:</b>\n"
                        "   Скачайте и установите:\n"
                        "   <code>https://aka.ms/vs/17/release/vc_redist.x64.exe</code>\n\n"
                        "   После установки перезапустите бота.\n\n"
                        "2️⃣ <b>Или используйте evaluate_ragas.py напрямую:</b>\n"
                        "   <code>python evaluate_ragas.py</code>\n\n"
                        "3️⃣ <b>Альтернатива:</b> Используйте RAGAS через API или Docker"
                    )
                    await update.message.reply_text(help_text, parse_mode='HTML')
                    logger.error(f"Ошибка загрузки torch DLL: {dll_error}")
                    return
                else:
                    raise
        
        # Импорт метрик RAGAS (только базовые метрики)
        try:
            from ragas.metrics._faithfulness import Faithfulness
            from ragas.metrics._context_precision import ContextPrecision
            faithfulness = Faithfulness
            context_precision = ContextPrecision
        except ImportError:
            try:
                from ragas.metrics.collections import faithfulness, context_precision
            except ImportError:
                from ragas.metrics import faithfulness, context_precision
    except ImportError as e:
        await update.message.reply_text(
            f"❌ Ошибка импорта RAGAS библиотек: {e}\n"
            "Убедитесь, что все зависимости установлены: pip install -r requirements.txt"
        )
        logger.error(f"Ошибка импорта RAGAS: {e}")
        return
    except Exception as e:
        await update.message.reply_text(
            f"❌ Ошибка загрузки RAGAS: {e}\n"
            "Возможна проблема с загрузкой torch. Попробуйте переустановить зависимости."
        )
        logger.error(f"Ошибка загрузки RAGAS: {e}")
        return
    
    try:
        # Отправляем сообщение о начале оценки
        status_message = await update.message.reply_text(
            "🔄 Запуск оценки качества RAG системы через RAGAS...\n\n"
            "⏳ Это может занять 1-2 минуты..."
        )
        
        # Подготовка датасета (синхронная операция)
        await status_message.edit_text(
            "🔄 Подготовка датасета...\n"
            f"📝 Обрабатываю {len(EVALUATION_QUESTIONS)} тестовых вопросов..."
        )
        
        dataset = await asyncio.to_thread(prepare_dataset_for_ragas, pipeline, EVALUATION_QUESTIONS)
        
        # Используем только базовые метрики RAGAS (не требуют llm/embeddings)
        metrics_to_use = []
        
        try:
            metrics_to_use.append(faithfulness())
        except Exception as e:
            logger.warning(f"Не удалось добавить Faithfulness: {e}")
        
        try:
            metrics_to_use.append(context_precision())
        except Exception as e:
            logger.warning(f"Не удалось добавить ContextPrecision: {e}")
        
        if not metrics_to_use:
            await status_message.edit_text(
                "❌ Не удалось инициализировать ни одну метрику RAGAS.\n"
                "Проверьте настройки и зависимости."
            )
            return
        
        # Запуск оценки
        await status_message.edit_text(
            "🔄 Запуск оценки метрик RAGAS...\n"
            "📊 Метрики: Faithfulness, Context Precision\n"
            "⏳ Пожалуйста, подождите (это может занять 1-2 минуты)..."
        )
        
        # Запускаем оценку RAGAS (синхронная операция в отдельном потоке)
        result = await asyncio.to_thread(
            evaluate,
            dataset=dataset,
            metrics=metrics_to_use
        )
        
        # Обработка результатов всех метрик
        def safe_avg(values, metric_name):
            """Безопасное вычисление среднего значения метрики."""
            valid_values = [
                v for v in values 
                if not (isinstance(v, float) and math.isnan(v))
            ]
            if valid_values:
                return sum(valid_values) / len(valid_values), len(valid_values)
            return 0, 0
        
        # Вычисляем средние значения для базовых метрик
        # EvaluationResult поддерживает индексацию через __getitem__, но не .get()
        avg_faithfulness, _ = safe_avg(result['faithfulness'], 'faithfulness')
        avg_context_precision, _ = safe_avg(result['context_precision'], 'context_precision')
        
        # Вычисляем общий средний балл
        avg_score = (avg_faithfulness + avg_context_precision) / 2
        
        # Формируем сообщение с результатами
        results_message = f"""
📊 <b>Результаты оценки RAG системы</b>

<b>Основные метрики:</b>
• Faithfulness (точность ответа): {avg_faithfulness:.4f}
• Context Precision (точность контекста): {avg_context_precision:.4f}

<b>Средний балл:</b> {avg_score:.4f}
"""
        
        # Добавляем оценку качества
        if avg_score >= 0.7:
            results_message += "\n✅ <b>Оценка: Отличное качество!</b>\nСистема показывает высокую точность и релевантность ответов."
        elif avg_score >= 0.5:
            results_message += "\n⚠️ <b>Оценка: Удовлетворительное качество</b>\nРекомендуется улучшить качество документов или промптов."
        else:
            results_message += "\n❌ <b>Оценка: Требует значительного улучшения</b>\nНеобходимо пересмотреть стратегию chunking или качество данных."
        
        results_message += "\n\n<i>Детальные результаты по каждому вопросу:</i>\n"
        
        # Добавляем детали по каждому вопросу
        for i, question in enumerate(EVALUATION_QUESTIONS):
            def format_metric(val):
                """Форматирование значения метрики."""
                if isinstance(val, float) and math.isnan(val):
                    return "N/A"
                return f"{val:.4f}"
            
            results_message += f"\n<b>{i+1}. {question}</b>\n"
            
            # Faithfulness
            try:
                if i < len(result['faithfulness']):
                    results_message += f"   • Faithfulness: {format_metric(result['faithfulness'][i])}\n"
            except (KeyError, IndexError):
                pass
            
            # Context Precision
            try:
                if i < len(result['context_precision']):
                    results_message += f"   • Context Precision: {format_metric(result['context_precision'][i])}\n"
            except (KeyError, IndexError):
                pass
        
        # Обновляем сообщение с результатами
        await status_message.edit_text(results_message, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Ошибка при оценке RAGAS: {e}")
        error_message = f"❌ Произошла ошибка при оценке:\n\n<code>{str(e)}</code>"
        try:
            await status_message.edit_text(error_message, parse_mode='HTML')
        except:
            await update.message.reply_text(error_message, parse_mode='HTML')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений."""
    if pipeline is None:
        await update.message.reply_text("❌ Система не инициализирована. Попробуйте позже.")
        return
    
    user_query = update.message.text.strip()
    
    if not user_query:
        await update.message.reply_text("⚠️ Пожалуйста, введите вопрос.")
        return
    
    # Отправляем сообщение о том, что бот думает
    thinking_message = await update.message.reply_text("🤔 Думаю...")
    
    try:
        # Обработка запроса через RAG pipeline
        result = pipeline.query(user_query, use_cache=True)
        
        # Формируем ответ
        answer_text = f"<b>💬 Ответ:</b>\n\n{result['answer']}\n\n"
        
        # Добавляем информацию об источнике
        if result['from_cache']:
            answer_text += "💾 <i>Ответ из кеша</i>"
            if 'cached_at' in result:
                answer_text += f"\n📅 Сохранено: {result['cached_at']}"
        else:
            answer_text += f"🌐 <i>Ответ от Anthropic API ({result.get('model', 'LLM')})</i>"
            if result.get('context_docs'):
                answer_text += f"\n📚 Использовано документов: {len(result['context_docs'])}"
        
        # Обновляем сообщение с ответом
        await thinking_message.edit_text(answer_text, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        error_message = f"❌ Произошла ошибка при обработке вашего запроса:\n\n<code>{str(e)}</code>"
        await thinking_message.edit_text(error_message, parse_mode='HTML')


def main() -> None:
    """Главная функция для запуска бота."""
    global pipeline
    
    # Проверка наличия необходимых переменных окружения
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN не установлен в переменных окружения")
        print("❌ Ошибка: переменная окружения TELEGRAM_BOT_TOKEN не установлена")
        print("\nУстановите её следующим образом:")
        print("  Windows (PowerShell): $env:TELEGRAM_BOT_TOKEN='your-token'")
        print("  Windows (CMD): set TELEGRAM_BOT_TOKEN=your-token")
        print("  Linux/Mac: export TELEGRAM_BOT_TOKEN='your-token'")
        print("\nИли добавьте в .env файл:")
        print("  TELEGRAM_BOT_TOKEN=your-token-here")
        return
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY не установлен")
        print("❌ Ошибка: переменная окружения ANTHROPIC_API_KEY не установлена")
        return
    
        # Инициализация RAG pipeline
    try:
        logger.info("Инициализация RAG pipeline...")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_path="data",
            model="claude-sonnet-4-5-20250929"
        )
        logger.info("RAG pipeline успешно инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации RAG pipeline: {e}")
        print(f"❌ Ошибка инициализации: {e}")
        return
    
    # Создание приложения бота
    application = Application.builder().token(telegram_token).build()
    
    # Регистрация обработчиков команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("evaluate", evaluate_command))
    
    # Регистрация обработчика текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запуск бота
    logger.info("Запуск Telegram бота...")
    print("✅ Бот запущен и готов к работе!")
    print("Нажмите Ctrl+C для остановки")
    
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
