"""
Module de nettoyage des fichiers temporaires et de sauvegarde de l'acronyme
client.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""
# imports
import os
from src.config import CLEANUP_TEMP_FILES


def cleanup_temp_files(temp_dir):
    """
    Clean all temporary files in the specified directory if the cleanup
    flag is set.
    """
    if CLEANUP_TEMP_FILES and os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)


def save_client_acronym(output_log_dir, client_acronym):
    """
    Save the client acronym to a text file in the specified output log
    directory.
    """
    with open(
        os.path.join(output_log_dir, "client_acronym.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(client_acronym)
