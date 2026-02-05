"""
Module de traitement des fichiers et des données extraites.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""
# Imports
import src.ocr_utils as func
from src.config import pattern


def process_files_and_data(
        text_extracted,
        data_dir,
        output_folder_dir, 
        output_json_dir, client_acronym
    ):
    """
    Copy files with mapping and group data by chimney name.
    """
    # copy files with mapping and import JSON file
    func.print_step(2, "Copie des fichiers avec mapping")
    key_info_file, mapping_file = func.copy_files_with_mapping(
        text_extracted,
        pattern,
        data_dir,
        output_folder_dir,
        output_json_dir,
        client_acronym
    )
    key_info_dict = func.import_json_to_text(
        output_json_dir,
        input_file=key_info_file
    )

    # group data by chimney name and save to JSON
    func.print_step(3, "Groupement des données par nom de cheminée")
    data_per_chimney = func.group_by_chimney_name(key_info_dict, pattern)
    func.save_to_json(
        data_per_chimney, output_json_dir, "data_per_chimney.json"
    )

    return key_info_dict, data_per_chimney


def get_client_name(key_info_dict):
    """
    Extract the main client name from the key information dictionary.
    """
    # get client name counts and determine the most frequent one
    func.print_step(4, "Récupération du nom du client")
    client_names = func.get_client_name_counts(key_info_dict)
    client_name = max(client_names, key=client_names.get)
    print(f"\nClient name: {client_name}")
    return client_name
