## Поддерживаемые ключи
| Ключ | Тип | Описание |
|------|-----|---------|
| `--config` | str | Путь к JSON-файлу конфигурации |
| `--mode` | SINGLE/GRID | Режим работы |
| `--input, -i` | str | Файл или директория |
| `--output, -o` | str | Директория результатов |
| `--grid` | int int | Размер сетки (строки столбцы) |
| `--step-mm` | float | Шаг сетки в мм |
| `--find-axes` | флаг | Поиск систем координат |
| `--show` | флаг | Показывать графики |
| `--debug` | флаг | Сохранять промежуточные шаги |

## Примеры запуска
```python
  # Одиночный крестик (одно фото):
  python cli.py --mode SINGLE --input data/photo.jpg --output results/

  # Одиночные крестики (вся папка):
  python cli.py --mode SINGLE --input data/crosses/ --output results/

  # Сетка 9x11 (одно фото):
  python cli.py --mode GRID --input data/grid.jpg --output results/ --grid 9 11

  # Через конфиг JSON:
  python cli.py --config config.json

  # С отображением графиков:
  python cli.py --mode SINGLE --input data/ --output results/ --show

  # Режим Debug (сохраняет промежуточные шаги):
  python cli.py --mode SINGLE --input data/ --output results/ --debug
```