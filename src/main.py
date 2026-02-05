"""
Ce fichier contient le script principal pour le projet OCR.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""
# # Import necessary libraries
import os
import sys
import warnings
import time

# Import custom OCR utilities and configurations
import src.ocr_utils as func
import src.config as config
from src.pipeline.setup import (
    setup_directories,
    print_banner_and_versions,
    get_user_inputs,
    get_files_by_subdir
)
from src.pipeline.extraction import extract_text_from_images
from src.pipeline.processing import process_files_and_data, get_client_name
from src.pipeline.reporting import generate_word_report
from src.pipeline.database import insert_into_database
from src.pipeline.backup import create_backups
from src.pipeline.cleanup import cleanup_temp_files, save_client_acronym


def main():
    """Main function to execute the OCR pipeline."""

    ################### SETUP ##########################
    # Get project directory (assumed to be parent of src)
    project_dir = os.getcwd().split("\\src")[0]

    # Setup directories
    dirs = setup_directories(project_dir)

    # Ignore specific warnings
    warnings.filterwarnings("ignore", message=".*pin_memory.*")

    # Redirect all prints to a log file
    if config.LOG_TO_FILE:
        sys.stdout = func.Logger(
            os.path.join(dirs["output_log_dir"], "process_log.txt")
        )

    # Print banner and versions
    print_banner_and_versions()

    # if logging to file, temporarily restore stdout to get user inputs
    if config.LOG_TO_FILE:
        old_stdout = sys.stdout
        sys.stdout = sys.__stdout__

    # Get user inputs
    (
        client_acronym,
        date_mesure,
        folder_ignored,
        folder_ignored_dir
    ) = get_user_inputs(dirs["data_dir"])

    # Restore stdout to log file
    if config.LOG_TO_FILE:
        sys.stdout = old_stdout

    # Print the inputs
    print("\nInputs:")
    print("client_acronym     : " + client_acronym)
    print("date_mesure        : " + date_mesure)
    print("folder_ignored_dir : " + folder_ignored_dir)

    # Start the timer
    start_time = time.time()

    # Get files by subdirectory
    files_by_subdir = get_files_by_subdir(dirs["data_dir"])
    func.save_to_json(
        files_by_subdir, dirs["output_json_dir"], "files_by_subdir.json"
    )

    ################ TEXT EXTRACTION  ##############

    # TEXT EXTRACTION
    text_extracted = extract_text_from_images(
        dirs["data_dir"],
        dirs["output_json_dir"],
        dirs["temp_dir"],
        files_by_subdir,
        folder_ignored,
        folder_ignored_dir,
        start_time
    )

    ############### DATA PROCESSING #################

    # PROCESS FILES AND DATA
    key_info_dict, data_per_chimney = process_files_and_data(
        text_extracted,
        dirs["data_dir"],
        dirs["output_folder_dir"],
        dirs["output_json_dir"],
        client_acronym
    )

    # GET CLIENT NAME
    client_name = get_client_name(key_info_dict)

    ############### REPORTING WORD #############

    # WORD REPORT if needed
    if config.GENERATE_WORD_REPORT:
        output_word_report_path = generate_word_report(
            data_per_chimney,
            dirs["data_dir"],
            dirs["output_dir"],
            dirs["temp_dir"],
            client_name,
            files_by_subdir,
            date_mesure,
            dirs["logo_dir"]
        )
    else:
        output_word_report_path = None
        print("\nGénération du rapport Word désactivée par le flag.")

    ############### DATABASE #############

    # DATABASE INSERTION if needed
    if config.INSERT_IN_DATABASE:
        bdd_path = insert_into_database(
            dirs["bdd_dir"],
            client_acronym,
            client_name,
            data_per_chimney,
            date_mesure
        )
    else:
        bdd_path = None
        print("\nInsertion dans la base de données désactivée par le flag.")

    ############### BACKUPS #############

    # BACKUPS if needed
    if config.IS_BACKUP_CREATED:
        create_backups(
            bdd_path,
            dirs["bdd_backup_dir"],
            output_word_report_path,
            dirs["word_backup_dir"],
            dirs["output_json_dir"],
            dirs["json_backup_dir"],
            client_acronym
        )
    else:
        print("\nCréation des backups désactivée par le flag.")

    ############### CLEANUP #############

    # cleanup temporary files
    cleanup_temp_files(dirs["temp_dir"])

    # Save client acronym
    save_client_acronym(dirs["output_log_dir"], client_acronym)

    # Print end message
    print("\n" + "*" * 75)
    print("==> FIN DU SCRIPT PRINCIPAL <==")
    print("*" * 75)
    func.calculate_duration(start_time)
    print(
        "\nAttendre encore 2 secondes pour s'assurer que les "
        "opérations de backup soient terminées..."
    )

# Execute main function
if __name__ == "__main__":
    main()
