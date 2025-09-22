# Optical Circuit Merger

Простое веб-приложение для объединения нескольких файлов (Excel/CSV) в один общий Excel с итоговыми «оптическими схемами».  

---

## 🚀 Быстрый старт (через Docker)

1. **Установи Docker**  
   [Инструкция по установке](https://docs.docker.com/get-docker/)

2. **Клонируй репозиторий**  
   ```bash
   git clone https://github.com/zetsudan/optical-circuit-merger
   cd optical-circuit-merger
   ```

3. **Собери контейнер**  
   ```bash
   docker build -t optical-circuit-merger:latest .
   ```

4. **Запусти контейнер**  
   ```bash
   docker run -d \
     --name optical-circuit-merger \
     -p 8000:8000 \
     --restart unless-stopped \
     optical-circuit-merger:latest
   ```

---

## 🌐 Использование

После запуска приложение будет доступно по адресу:  
👉 IP:8000

- На главной странице можно загрузить несколько файлов для объединения.  
- Результат автоматически скачивается в виде Excel-файла.  
- Имя файла генерируется в формате:  
  ```
  optical_circuits_merge_HHMM_DDMMYYYY.xlsx
  ```

---

## 🛠️ Для разработчиков (локальный запуск)

Если нужно запустить без Docker:

```bash
git clone https://github.com/zetsudan/optical-circuit-merger
cd optical-circuit-merger

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📌 Примечания

- Порт по умолчанию: **8000**  
- Чтобы остановить контейнер:  
  ```bash
  docker stop optical-circuit-merger
  ```
- Чтобы удалить контейнер:  
  ```bash
  docker rm -f optical-circuit-merger
  ```

---

## 📄 Лицензия
MIT
