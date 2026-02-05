"""
Module de gestion de la base de données pour l'insertion des données mesurées.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""
# imports
import src.ocr_utils as func


def insert_into_database(
    bdd_dir,
    client_acronym,
    client_name,
    data_per_chimney,
    date_mesure
):
    """
    Insert data into the database for a given client and measurement date.
    """
    # Initialization of the database path
    bdd_path = None

    # Print step
    func.print_step(6, "Insertion des données dans la base de données")

    # Create database and tables if they do not exist
    bdd_path = func.create_database_and_tables(bdd_dir)

    # Insert client, chimney, and measurement data into the database
    func.insert_client_into_db(bdd_path, client_acronym, client_name)
    func.insert_cheminee_into_db(bdd_path, client_acronym, data_per_chimney)
    func.insert_mesure_into_db(
        bdd_path,
        client_acronym,
        data_per_chimney,
        date_mesure
    )

    return bdd_path
