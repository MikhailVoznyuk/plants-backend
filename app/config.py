import os

DEVICE = os.getenv("DEVICE", "cpu")
THRESH_PLANT = float(os.getenv("THRESH_PLANT", "0.2"))
THRESH_DEFECT = float(os.getenv("THRESH_DEFECT", "0.25"))

PLANT_CLASSES = [x.strip() for x in os.getenv("PLANT_CLASSES", "tree,shrub").split(",") if x.strip()]
PLANT_COLORS = [x.strip() for x in os.getenv("PLANT_COLORS", "").split(",") if x.strip()]
DEFECT_COLORS = [x.strip() for x in os.getenv("DEFECT_COLORS", "").split(",") if x.strip()]

PLANT_SEG_WEIGHTS = os.getenv("PLANT_SEG_WEIGHTS", "").strip()
DEFECT_SEG_WEIGHTS = os.getenv("DEFECT_SEG_WEIGHTS", "").strip()
SPECIES_CLS_WEIGHTS = os.getenv("SPECIES_CLS_WEIGHTS", "").strip()
SPECIES_STUB_LABEL = os.getenv("SPECIES_STUB_LABEL", "unknown")

SEVERITY_RULES_CSV = os.getenv("SEVERITY_RULES_CSV", "").strip()
OUT_DIR = os.getenv("OUT_DIR", "/data/out")

# Фоллбеки и параметры порогов
PLANT_FALLBACK = os.getenv("PLANT_FALLBACK", "1") == "1"     # включить HSV fallback для растений
DEFECT_FALLBACK = os.getenv("DEFECT_FALLBACK", "0") == "1"   # включить "коричневые пятна" как дефекты (по желанию)

# HSV диапазон «зелени»
VEG_H_MIN = int(os.getenv("VEG_H_MIN", "30"))    # ~зеленый
VEG_H_MAX = int(os.getenv("VEG_H_MAX", "90"))
VEG_S_MIN = int(os.getenv("VEG_S_MIN", "40"))
VEG_V_MIN = int(os.getenv("VEG_V_MIN", "40"))

# HSV диапазон «бурого» для псевдо-дефектов
BRN_H_MIN = int(os.getenv("BRN_H_MIN", "10"))    # коричневый/рыжий
BRN_H_MAX = int(os.getenv("BRN_H_MAX", "25"))
BRN_S_MIN = int(os.getenv("BRN_S_MIN", "20"))
BRN_V_MAX = int(os.getenv("BRN_V_MAX", "180"))