"""
Fichier de configuration pour le projet OCR.

Ce fichier contient les paramètres de configuration pour le projet OCR, y
compris les chemins d'accès aux fichiers, les paramètres de traitement d'image
et les options de modèle.
Il est recommandé de ne pas modifier ce fichier à moins que vous ne sachiez ce
que vous faites.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Avril 2025
"""

# Parameters for image processing and OCR
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
