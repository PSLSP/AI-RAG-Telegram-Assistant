"""
Модуль работы с векторным хранилищем FAISS.
Обрабатывает загрузку документов, chunking и поиск по векторам.
"""

import faiss
import numpy as np
import pickle
from typing import List, Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Пытаемся загрузить из текущей директории
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе FAISS."""
    
    def __init__(self, collection_name: str = "rag_collection", persist_directory: str = "./faiss_db"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции
            persist_directory: директория для хранения данных
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Создаём директорию, если её нет
        os.makedirs(persist_directory, exist_ok=True)
        
        # Пути к файлам
        self.index_path = os.path.join(persist_directory, f"{collection_name}.index")
        self.metadata_path = os.path.join(persist_directory, f"{collection_name}.pkl")
        
        # Инициализация структур данных
        self.index = None
        self.documents = []  # Список текстов документов
        self.ids = []  # Список ID документов
        self.embedding_dim = None  # Размерность embeddings
        self.loaded_files_info = {}  # Информация о загруженных файлах {путь: время_модификации}
        
        # Загрузка существующего индекса или создание нового
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load_index()
            print(f"Коллекция '{collection_name}' загружена. Документов: {len(self.documents)}")
        else:
            print(f"Создана новая коллекция '{collection_name}'")
        
        # OpenAI клиент для создания embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _load_index(self):
        """Загрузка индекса и метаданных с диска."""
        try:
            # Загрузка FAISS индекса
            self.index = faiss.read_index(self.index_path)
            
            # Загрузка метаданных (документы и ID)
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.documents = metadata.get('documents', [])
                self.ids = metadata.get('ids', [])
                self.embedding_dim = metadata.get('embedding_dim', None)
                self.loaded_files_info = metadata.get('loaded_files_info', {})
            
            # Проверка согласованности
            if self.index.ntotal != len(self.documents):
                raise ValueError("Несоответствие между индексом и метаданными")
                
        except Exception as e:
            print(f"Ошибка загрузки индекса: {e}")
            # Создаём новый индекс
            self.index = None
            self.documents = []
            self.ids = []
            self.embedding_dim = None
            self.loaded_files_info = {}
    
    def _save_index(self):
        """Сохранение индекса и метаданных на диск."""
        if self.index is None:
            return
        
        try:
            # Сохранение FAISS индекса
            faiss.write_index(self.index, self.index_path)
            
            # Сохранение метаданных
            metadata = {
                'documents': self.documents,
                'ids': self.ids,
                'embedding_dim': self.embedding_dim,
                'loaded_files_info': getattr(self, 'loaded_files_info', {})
            }
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
        except Exception as e:
            print(f"Ошибка сохранения индекса: {e}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Нормализация векторов для cosine similarity.
        
        Args:
            vectors: массив векторов
            
        Returns:
            нормализованные векторы
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Избегаем деления на ноль
        return vectors / norms
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        import re
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _check_files_changed(self, file_paths: list) -> bool:
        """
        Проверка, изменились ли файлы с момента последней загрузки.
        
        Args:
            file_paths: список путей к файлам
            
        Returns:
            True, если файлы изменились или появились новые
        """
        loaded_files_info = getattr(self, 'loaded_files_info', {})
        if not loaded_files_info:
            return True  # Если нет информации о загруженных файлах, считаем что нужно загрузить
        
        current_files = set(file_paths)
        loaded_files = set(loaded_files_info.keys())
        
        # Проверяем, изменился ли набор файлов
        if current_files != loaded_files:
            print(f"Обнаружены изменения в наборе файлов (было: {len(loaded_files)}, стало: {len(current_files)})")
            return True
        
        # Проверяем даты модификации каждого файла
        for file_path in file_paths:
            try:
                current_mtime = os.path.getmtime(file_path)
                saved_mtime = loaded_files_info.get(file_path)
                
                if saved_mtime is None or current_mtime != saved_mtime:
                    print(f"Файл {os.path.basename(file_path)} был изменен")
                    return True
            except Exception as e:
                print(f"Ошибка при проверке файла {file_path}: {e}")
                return True
        
        return False
    
    def load_documents(self, data_path: str, force_reload: bool = False):
        """
        Загрузка документов из файла или папки в векторное хранилище.
        
        Args:
            data_path: путь к файлу с документами или папке с .txt файлами
            force_reload: принудительная перезагрузка, даже если файлы не изменились
        """
        # Проверка существования пути
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Путь {data_path} не найден")
        
        # Определяем, это файл или папка
        if os.path.isfile(data_path):
            # Если это файл, загружаем его
            file_paths = [data_path]
        elif os.path.isdir(data_path):
            # Если это папка, находим все .txt файлы
            file_paths = []
            for file_name in os.listdir(data_path):
                if file_name.lower().endswith('.txt'):
                    file_path = os.path.join(data_path, file_name)
                    if os.path.isfile(file_path):
                        file_paths.append(file_path)
            
            if not file_paths:
                raise FileNotFoundError(f"В папке {data_path} не найдено .txt файлов")
        else:
            raise ValueError(f"{data_path} не является ни файлом, ни папкой")
        
        # Проверяем, нужно ли перезагружать документы
        if not force_reload and len(self.documents) > 0:
            print(f"Проверка изменений файлов... (загружено файлов: {len(self.loaded_files_info)})")
            files_changed = self._check_files_changed(file_paths)
            if not files_changed:
                print("✓ Документы уже загружены и не изменились. Используется существующий индекс.")
                return
            else:
                print("⚠ Обнаружены изменения в файлах. Перезагружаем документы...")
                # Очищаем существующие данные
                self.index = None
                self.documents = []
                self.ids = []
                self.embedding_dim = None
                self.loaded_files_info = {}
        
        print(f"Найдено {len(file_paths)} .txt файлов в папке {data_path}")
        
        # Читаем все файлы и объединяем их содержимое
        all_texts = []
        files_info = {}  # Сохраняем информацию о загруженных файлах
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                    if file_text.strip():  # Пропускаем пустые файлы
                        # Добавляем информацию о файле в начало текста
                        file_name = os.path.basename(file_path)
                        all_texts.append(f"\n\n[Документ: {file_name}]\n\n{file_text}")
                        
                        # Сохраняем время модификации файла
                        files_info[file_path] = os.path.getmtime(file_path)
                        
                        print(f"Загружен файл: {file_name} ({len(file_text)} символов)")
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")
                continue
        
        if not all_texts:
            raise ValueError("Не удалось загрузить ни одного документа")
        
        # Объединяем все тексты
        combined_text = "\n".join(all_texts)
        print(f"Общий размер загруженных документов: {len(combined_text)} символов")
        
        # Разбиение на чанки
        chunks = self._chunk_text(combined_text)
        print(f"Текст разбит на {len(chunks)} чанков")
        
        # Создание embeddings и подготовка данных
        documents = []
        ids = []
        embeddings = []
        
        for i, chunk in enumerate(chunks):
            # Создание embedding через OpenAI
            embedding = self._create_embedding(chunk)
            
            documents.append(chunk)
            ids.append(f"doc_{i}")
            embeddings.append(embedding)
            
            if (i + 1) % 10 == 0:
                print(f"Обработано {i + 1}/{len(chunks)} чанков")
        
        # Определение размерности embeddings
        if embeddings:
            self.embedding_dim = len(embeddings[0])
            
            # Преобразование в numpy массив
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Нормализация для cosine similarity
            embeddings_array = self._normalize_vectors(embeddings_array)
            
            # Создание FAISS индекса
            # Используем IndexFlatIP (Inner Product) для cosine similarity с нормализованными векторами
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Добавление векторов в индекс
            self.index.add(embeddings_array)
            
            # Сохранение документов и ID
            self.documents = documents
            self.ids = ids
            self.loaded_files_info = files_info  # Сохраняем информацию о загруженных файлах
            
            # Сохранение на диск
            self._save_index()
            
            print(f"Загружено {len(chunks)} чанков из {len(file_paths)} файлов в коллекцию '{self.collection_name}'")
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        Создание векторного представления текста через OpenAI.
        
        Args:
            text: текст для векторизации
            
        Returns:
            вектор embeddings
        """
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            
        Returns:
            список документов с метаданными
        """
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Создание embedding для запроса
        query_embedding = self._create_embedding(query)
        
        # Преобразование в numpy массив и нормализация
        query_vector = np.array([query_embedding]).astype('float32')
        query_vector = self._normalize_vectors(query_vector)
        
        # Поиск в FAISS индексе
        # top_k не может быть больше количества документов
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        # Форматирование результатов
        documents = []
        if len(indices[0]) > 0:
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    # Для cosine similarity: distance = 1 - similarity
                    # FAISS IndexFlatIP возвращает inner product (cosine similarity для нормализованных векторов)
                    similarity = distances[0][i]
                    distance = 1.0 - similarity  # Преобразуем similarity в distance
                    
                    documents.append({
                        'id': self.ids[idx],
                        'text': self.documents[idx],
                        'distance': float(distance),
                        'similarity': float(similarity)
                    })
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        count = len(self.documents) if self.documents else 0
        return {
            'name': self.collection_name,
            'count': count,
            'persist_directory': self.persist_directory,
            'embedding_dim': self.embedding_dim
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: установите переменную окружения OPENAI_API_KEY")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов
    if os.path.exists("data/docs.txt"):
        vector_store.load_documents("data/docs.txt")
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']:.4f}, Similarity: {doc['similarity']:.4f}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")
