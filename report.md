# Порівняння Fat vs Slim — inference MobileNetV2

Цифри нижче — з реального локального білду на Linux x86_64
(Docker 29.4.0, overlayfs). Перезапустіть команди після власного
`docker build`, щоб оновити їх.

## Як зібрано числа

```bash
docker build -f Dockerfile.fat  -t mlops-lesson3:fat  .
docker build -f Dockerfile.slim -t mlops-lesson3:slim .

docker images mlops-lesson3
docker history mlops-lesson3:fat  --no-trunc --format 'table {{.Size}}\t{{.CreatedBy}}'
docker history mlops-lesson3:slim --no-trunc --format 'table {{.Size}}\t{{.CreatedBy}}'
```

Обидва образи дають однаковий результат на `samples/dog.jpg`:

```
1. Samoyed       39.75%
2. Pomeranian     4.31%
3. Arctic fox     3.60%
```

## Розмір і кількість шарів

| Образ                | База               | Розмір  | Шари |
|----------------------|--------------------|---------|------|
| `mlops-lesson3:fat`  | `ubuntu:22.04`     | 14.9 ГБ | 12   |
| `mlops-lesson3:slim` | `python:3.11-slim` | 1.48 ГБ | 18   |
| **Зменшення**        |                    | **≈ 10×** | +6 |

Fat-образ ~10× більший, навіть попри те, що slim-образ має
*більше* шарів — кількість шарів переважно успадковується від
базового образу, а розмір визначається тим, *що* в цих шарах
лежить.

### Куди йдуть байти

Fat-образ, найбільші внески (з `docker history`):

| Розмір   | Походження |
|----------|------------|
| 8.24 ГБ  | `pip3 install torch==2.3.1 torchvision==0.18.1 pillow Django` (CUDA-колеса + кеш pip) |
|  521 МБ  | `apt-get install build-essential python3-dev git curl wget vim nano htop ...` |
|   88 МБ  | rootfs Ubuntu 22.04 |
|   14 МБ  | `model.pt` + `imagenet_classes.json` + `inference.py` |

Slim-образ, найбільші внески:

| Розмір   | Походження |
|----------|------------|
| 1.0  ГБ  | `/opt/venv` (CPU-only `torch+cpu`, `torchvision+cpu`, `pillow`), скопійований з builder |
|  138 МБ  | Шари встановлення Python 3.11 у `python:3.11-slim` |
|   87 МБ  | rootfs `debian:trixie-slim` |
|   14 МБ  | `model.pt` + `imagenet_classes.json` + `inference.py` |

## Що всередині (і чого там бути не повинно)

### Fat-образ — `Dockerfile.fat`

- База: повноцінна **Ubuntu 22.04** (≈ 80 МБ ще до встановлення чогось).
- Системні пакети, що залишаються у фінальному образі:
  `build-essential gcc g++ make cmake git curl wget vim nano htop tree unzip`
  плюс dev-заголовки (`python3-dev`, `libjpeg-dev`, `zlib1g-dev`,
  `libpng-dev`). Жоден з них не потрібен під час inference.
- Python-колеса встановлюються з **дефолтного індексу**, тож
  `torch==2.3.1` тягне CUDA-варіант (`nvidia-cuda-*`, `nvidia-cudnn`,
  `nvidia-cublas`, …) — приблизно 2 ГБ GPU-runtime бібліотек, які
  CPU-контейнер ніколи не завантажить.
- `pip3 install` запущений з дефолтним кешем, тож `~/.cache/pip`
  теж залишається запеченим у шарі.
- Зайві Python-залежності (`Django`) тягнуться без жодної потреби.

### Slim-образ — `Dockerfile.slim`

- База: **`python:3.11-slim`** (~45 МБ) — Debian slim, лише
  необхідне.
- **Multi-stage збірка**: `build-essential` живе в builder-стадії
  і ніколи не потрапляє у runtime-образ.
- CPU-only PyTorch: `torch==2.3.1+cpu` із
  `https://download.pytorch.org/whl/cpu` — повністю прибирає
  CUDA-стек.
- `PIP_NO_CACHE_DIR=1` тримає кеш wheel-ів поза шарами.
- Залежності лежать у virtualenv-і `/opt/venv`, який копіюється
  одним COPY-шаром.
- Запускається від ім'я не-root користувача `app`; у `/app` лежать
  лише `model.pt`, `imagenet_classes.json` та `inference.py`.

## Проблеми «жирного» образу

1. **Bandwidth і cold start.** Качати ~2 ГБ на кожен ноду / CI-runner —
   повільно та дорого, і більшість цього об'єму — невикористаний
   CUDA-стек.
2. **Поверхня атаки.** Компілятори, `git`, `curl`, `wget`, редактори
   й адмін-інструменти всі стають поверхнею атаки всередині
   контейнера.
3. **Відтворюваність.** Дефолтний torch wheel змінює свій варіант
   залежно від архітектури хоста / pip-резолвера; явне закріплення
   `+cpu` (slim-образ) усуває цю неоднозначність.
4. **Кеш-тершинг.** Великі блоки `apt-get install` і `pip install`
   живуть в окремих шарах, тому одна зміна в рядку викликає
   пересборку на кілька гігабайтів.
5. **Користувач root** за замовчуванням — погана практика для
   будь-якого сервісу.

## Подальші ідеї оптимізації

- **Distroless / scratch runtime.** Замінити `python:3.11-slim` на
  `gcr.io/distroless/python3-debian12` (або взагалі на
  `scratch`-білд із статичним libtorch), щоб прибрати з runtime
  менеджер пакетів і shell.
- **Тільки `torchscript-lite` / `libtorch`.** TorchScript inference
  не потребує повного Python-пакета `torch` — C++ runtime навколо
  `libtorch` завантажує `model.pt` напряму. Образ зменшується до
  ~150 МБ.
- **ONNX / OpenVINO / TensorRT.** Сконвертувати `.pt` у ONNX і
  обслуговувати через `onnxruntime` (~50 МБ wheel), додатково
  отримуючи крос-платформне прискорення.
- **Квантизація.** `torch.quantization.quantize_dynamic` зменшує
  розмір моделі в 3-4 рази та прискорює CPU-inference.
- **`--platform linux/amd64`** явно + `docker buildx` з
  cache mounts (`--mount=type=cache,target=/root/.cache/pip`),
  щоб не зберігати кеш wheel-ів у жодному шарі і водночас
  отримувати кеш при повторних білдах.
- **Прибрати pip** з runtime-образу через `pip install --target` і
  копіювати лише `site-packages`, або використати `uv pip` для
  швидшої установки.
- **Сканування образу.** Запустити `docker scout cves mlops-lesson3:slim`
  і `dive mlops-lesson3:slim`, щоб знайти зайві файли та
  непатчені CVE.
