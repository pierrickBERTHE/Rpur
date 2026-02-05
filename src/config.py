"""
Fichier de configuration pour le projet OCR.

Ce fichier contient les paramètres et les drapeaux utilisés dans le projet OCR.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""

# Define some flags
IS_CORRECT_TEXT_FRENCH = False
USE_GPU_FOR_OCR = False
GENERATE_WORD_REPORT = True
INSERT_IN_DATABASE = True
CLEANUP_TEMP_FILES = True
LOG_TO_FILE = True
IS_BACKUP_CREATED = True

# Parameters for image processing and OCR (adjusted since POC notebook)
best_params = {
    'adjust_contrast': 0.5,
    'batch_size': 1,
    'decoder': "wordbeamsearch",
    'scale_percent': 25,
    'worker': 0
}

# Pattern to search for ("a(1)", etc.), exclude if unit of measurement follows
pattern = (
    r"([a-zA-Z]+)(\d+)"
    r"(?!\s*("
    r"cm|cm2|mm|mm2|dm|dm2|km|km2|hm|µm|nm|pm|in|ft|yd|mi|kg|mg|ml|cl|dl|hl|"
    r"ms|µs|ns|ps|min|°c|°f|pa|kpa|mpa|bar|mb|db|mv|kv|ma|ka|"
    r"kw|mw|gw|hz|khz|mhz|ghz|kb|mb|gb|tb|pb|go|ko|mo"
    r")\b)"
)
