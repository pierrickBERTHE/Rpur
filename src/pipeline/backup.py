""" 
Module de gestion des sauvegardes.

Ce module contient des fonctions pour créer des sauvegardes de la base de données, des rapports Word et des fichiers JSON.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""
# Imports
import src.ocr_utils as func


def create_backups(
        bdd_path,
        bdd_backup_dir,
        output_word_report_path,
        word_backup_dir,
        output_json_dir,
        json_backup_dir,
        client_acronym
    ):
    """
    Create save backups of the database, Word report, and JSON files.
    """
    # Print step header
    func.print_step(7, "Création des sauvegardes")

    # Database backup if path is provided
    if bdd_path:
        func.backup_database(bdd_path, backup_dir=bdd_backup_dir)
    else:
        print("Aucune base de données trouvée, sauvegarde ignorée.")

    # Backup Word report if it exists
    if output_word_report_path:
        func.backup_word_report(
            output_word_report_path,
            word_backup_dir,
            client_acronym
        )
    else:
        print("Aucun rapport Word trouvé, sauvegarde ignorée.")

    # Backup JSON files
    func.backup_json(
        output_json_dir,
        json_backup_dir,
        client_acronym
    )
