# Tree Health Inference Service (Docker, FastAPI) — с авто-правилами

A → D → B → ROI → link → C → эвристики → визуализация, JSON, CSV.

## Запуск
CPU:
```bash
docker compose -f docker-compose.cpu.yml up --build
```
GPU:
```bash
docker compose -f docker-compose.gpu.yml up --build
```
Инференс:
```bash
curl -s -X POST http://localhost:8000/infer -F "file=@/path/to/photo.jpg" | jq
```

## Правила тяжести и health без CSV
Если `SEVERITY_RULES_CSV` не указан или файл отсутствует, сервис сам:
- считает `area_ratio` дефекта (площадь дефекта / площадь связанного растения);
- строит пороги по каждому классу дефекта как медиана и 85-й перцентиль `area_ratio`;
- присваивает `severity`: low / medium / high по этим порогам;
- повышает уровень для `fungus`/`pests`, если `dry_ratio > 0.45`;
- повышает уровень для `crack`/`cavity`/`mech_damage`, если `tilt_deg > 20`.

Далее для дерева вычисляется `health_score` (0..100):
- штрафы = сумма по дефектам `weight[class] * area_ratio * mult[severity]`;
  где `weight`: cavity 60, crack 45, mech_damage 35, fungus 30, pests 25 (дефолт 25),
  `mult`: low 0.7, medium 1.0, high 1.5;
- плюс 0.30 × `dry_ratio` × 100; плюс 0.8 × max(0, `tilt_deg` − 10).
- `health_grade`: good ≥80, fair 60..79, poor 40..59, critical <40.

Если `SEVERITY_RULES_CSV` задан, то значения оттуда имеют приоритет.

## Замена весов
Правь `.env`:
```
PLANT_SEG_WEIGHTS=/weights/plant_seg.pt
DEFECT_SEG_WEIGHTS=/weights/defect_seg.pt
SPECIES_CLS_WEIGHTS=/weights/species_cls.onnx
DEVICE=cuda:0
```
Папки `./weights`, `./data`, `./rules` монтируются в контейнер.