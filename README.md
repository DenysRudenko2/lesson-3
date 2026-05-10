# Урок 3 — Bash + Docker для ML inference

Невеликий класифікатор ImageNet на базі MobileNetV2, упакований
у два Docker-образи (fat і slim), плюс Bash-скрипт первинного
налаштування середовища.

## Структура

```
lesson-3/
├── install_dev_tools.sh      # ідемпотентний інсталятор (Docker, Python, ML-залежності)
├── export_model.py           # експортує MobileNetV2 у TorchScript model.pt
├── inference.py              # top-3 inference для зображення або папки
├── requirements.txt          # Python-залежності для локальних запусків
├── Dockerfile.fat            # ubuntu:22.04 + повний torch з CUDA (важкий)
├── Dockerfile.slim           # python:3.11-slim + CPU torch (multi-stage)
├── report.md                 # детальний аналіз fat vs slim
├── comparison.txt            # короткий конспект порівняння
├── model.pt                  # генерується через export_model.py
└── imagenet_classes.json     # генерується через export_model.py
```

## 1. Підготовка середовища

```bash
chmod +x install_dev_tools.sh
./install_dev_tools.sh
```

Скрипт є ідемпотентним: він повторно перевіряє кожну залежність
(Docker, Docker Compose, Python ≥ 3.9, pip, `torch`, `torchvision`,
`pillow`, `Django`) і встановлює лише те, чого не вистачає.
Увесь вивід дублюється у файл `install.log`.

## 2. Експорт TorchScript-моделі

`model.pt` створюється локально, щоб його можна було вкласти
в обидва образи:

```bash
python3 export_model.py
# -> model.pt  (~14 МБ)
# -> imagenet_classes.json
```

## 3. Локальний запуск inference (опційно)

```bash
python3 inference.py path/to/image.jpg
# або по всій папці із зображеннями
python3 inference.py ./samples/
```

Приклад виведення:

```
dog.jpg:
  1. Samoyed                         39.75%
  2. Pomeranian                       4.31%
  3. Arctic fox                       3.60%
```

## 4. Збірка обох Docker-образів

```bash
docker build -f Dockerfile.fat  -t mlops-lesson3:fat  .
docker build -f Dockerfile.slim -t mlops-lesson3:slim .

docker images mlops-lesson3
```

## 5. Запуск inference у контейнері

```bash
# Підмонтовуємо папку із зображеннями як /data і класифікуємо одне з них.
docker run --rm -v "$(pwd)/samples:/data" mlops-lesson3:slim /data/dog.jpg
docker run --rm -v "$(pwd)/samples:/data" mlops-lesson3:fat  /data/dog.jpg
```

`ENTRYPOINT` встановлено в `python inference.py`, тож все, що ви
додаєте після назви образу в `docker run`, передається як аргументи
для inference.py (шлях до зображення, `-k`, `--model`, `--classes`).

## 6. Порівняння образів

Дивіться [`report.md`](./report.md) — повний розбір (розміри, шари,
зайвий вміст, ідеї оптимізації) та [`comparison.txt`](./comparison.txt) —
коротку версію.

## Здача завдання

```bash
git checkout -b lesson-3
git add .
git commit -m "Add lesson-3: TorchScript model, Dockerfiles, report"
git push --set-upstream origin lesson-3
```
