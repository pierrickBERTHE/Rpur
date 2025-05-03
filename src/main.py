"""
Ce fichier contient le script principal pour le projet OCR.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Avril 2025
"""

from src.ocr_utils import process_images
from src.config import config

if __name__ == "__main__":
    process_images(
        input_folder=config["input_folder"],
        output_folder=config["output_folder"],
        batch_size=config["batch_size"]
    )
