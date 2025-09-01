"""
Ce fichier contient le script principal pour le projet OCR.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Août 2025
"""
# Import necessary libraries
import json
import os
import sys

import ocr_utils as func
from config import best_params, pattern
from tqdm import tqdm
import datetime
import time
import importlib.metadata
import PIL
import warnings

if __name__ == "__main__":

    ##################### DIRECTORIES SETUP #####################
    
    # Define some flags
    IS_CORRECT_TEXT_FRENCH = False
    USE_GPU_FOR_OCR = False
    GENERATE_WORD_REPORT = True
    INSERT_IN_DATABASE = True
    CLEANUP_TEMP_FILES = True
    LOG_TO_FILE = True
    IS_BACKUP_CREATED = True

    # Directories setup (in logical order)
    project_dir = os.getcwd().split("\\src")[0]

    # Input/output/data directories
    data_dir = os.path.join(project_dir, "data", "input", "source")
    output_dir = os.path.join(project_dir, "data", "output")
    output_json_dir = os.path.join(output_dir, "json")
    output_folder_dir = os.path.join(output_dir, "folder_output")
    output_log_dir = os.path.join(output_dir, "log")
    temp_dir = os.path.join(output_dir, "temp")

    # Other project directories
    logo_dir = os.path.join(project_dir, "image")
    bdd_dir = os.path.join(project_dir, "bdd")

    # Backup directories (outside project_dir)
    backup_dir = os.path.join(os.path.dirname(project_dir), "backup")
    bdd_backup_dir = os.path.join(backup_dir, "bdd")
    word_backup_dir = os.path.join(backup_dir, "word")

    # Check and create all directories if they do not exist
    func.check_and_create_directories(
        data_dir,
        output_dir,
        output_json_dir,
        output_folder_dir,
        output_log_dir,
        temp_dir,
        logo_dir,
        bdd_dir,
        backup_dir,
        bdd_backup_dir,
        word_backup_dir
    )

    # Ignore specific warnings
    warnings.filterwarnings("ignore", message=".*pin_memory.*")

    # Redirect all prints to a log file
    if LOG_TO_FILE:
        sys.stdout = func.Logger(
            os.path.join(output_log_dir, "process_log.txt")
        )

    # Print the start message
    print("""
    *****************************************
    *                                       *
    *   RRRRR    PPPPP    U   U    RRRRR    *
    *   R   R    P   P    U   U    R   R    *
    *   RRRRR    PPPPP    U   U    RRRRR    *
    *   R  R     P        U   U    R  R     *
    *   R   R    P         UUU     R   R    *
    *                                       *
    *****************************************
    """
    )

    # print the git version
    git_version = func.get_git_version()
    print(func.format_git_version(git_version))

    # Print all flags
    print("\nflags :")
    print("IS_CORRECT_TEXT_FRENCH :", IS_CORRECT_TEXT_FRENCH)
    print("USE_GPU_FOR_OCR        :", USE_GPU_FOR_OCR)
    print("GENERATE_WORD_REPORT   :", GENERATE_WORD_REPORT)
    print("INSERT_IN_DATABASE     :", INSERT_IN_DATABASE)
    print("CLEANUP_TEMP_FILES     :", CLEANUP_TEMP_FILES)
    print("LOG_TO_FILE            :", LOG_TO_FILE)
    print("IS_BACKUP_CREATED      :", IS_BACKUP_CREATED)

    ###################### PRINT LIBRAIRIES VERSIONS #####################

    print("\nInterpréteur python :")
    print("Python        : " + sys.version + "\n")

    print("Version des librairies utilisées :")
    print("Docx          : " + func.docx.__version__)
    print("Easyocr       : " + func.easyocr.__version__)
    print(
        "LanguageTool  : " + importlib.metadata.version("language-tool-python")
    )
    print("Numpy         : " + func.np.__version__)
    print("Pandas        : " + func.pd.__version__)
    print("Pillow        : " + PIL.__version__)
    print("Pytorch       : " + importlib.metadata.version("torch"))
    print("OpenCV        : " + func.cv2.__version__) #pylint: disable=no-member
    print("TQDM          : " + importlib.metadata.version("tqdm"))

    # Print time
    now = datetime.datetime.now().isoformat()
    print("\nCode lance le : " + now + "\n")

    ##################### INPUT USER #####################
    if LOG_TO_FILE:
        # Save the current stdout to restore it later
        old_stdout = sys.stdout
        sys.stdout = sys.__stdout__ 

    # Input from the user
    (
        client_acronym,
        date_mesure,
        folder_ignored,
        folder_ignored_dir
    ) = func.get_user_inputs(data_dir)

    if LOG_TO_FILE:
        # Activate the logger again
        sys.stdout = old_stdout

    # Print the inputs
    print("\nInputs:")
    print("client_acronym     : " + client_acronym)
    print("date_mesure        : " + date_mesure)
    print("folder_ignored_dir : " + folder_ignored_dir)

    # Start the timer
    start_time = time.time()

    ##################### TEXT EXTRACTION #####################
    # Print the step
    func.print_step(1, "Extraction du texte des images")

    # Text extracted file
    output_file_name="text_extracted.json"
    text_extracted_path = os.path.join(output_json_dir, output_file_name)

    # Create a dictionary to store files for each subdirectory
    subdir_list, files_by_subdir = [], {}
    files_by_subdir = func.get_files_by_subdir(data_dir)

    # Check if the output file exists, if not create it
    if not os.path.exists(text_extracted_path):
        text_extracted = {}

        # Extract text from the first X files in each subdirectory
        for subdir, files in tqdm(
            files_by_subdir.items(), desc="Analyse des dossiers"
        ):
            text_extracted[subdir] = {}

            # Skip the ignored folder with empty text
            if subdir == folder_ignored_dir or subdir == folder_ignored:
                print(f"\nDossier ignoré : '{subdir}'")
                for file in files:
                    text_extracted[subdir][file] = ""

            else:
                # All files in subdirectory
                for file in tqdm(
                    files[:], desc=f"{subdir} (processing)", leave=False
                    ):

                    # Get the image paths
                    image_path = os.path.join(data_dir, subdir, file)
                    image_processed_path = os.path.join(
                        temp_dir, "image_processed.jpg"
                    )
                    image_resized_path = os.path.join(
                        temp_dir, "resized_image.jpg"
                    )

                    # Process the image
                    image_processed = func.preprocess_black_text(
                        image_path, image_processed_path
                    )

                    # Resize the image
                    image_resized = func.resize_image(
                        image_processed_path,
                        image_resized_path,
                        scale_percent=best_params["scale_percent"]
                    )

                    # Extract text
                    text, duration = func.extract_text_easyocr(
                        image_resized_path,
                        batch_size=best_params["batch_size"],
                        decoder=best_params["decoder"],
                        adjust_contrast=best_params["adjust_contrast"],
                        worker=best_params["worker"],
                        gpu_state=USE_GPU_FOR_OCR
                    )

                    # Clean the french text
                    if IS_CORRECT_TEXT_FRENCH:
                        text = func.correct_text_french(text)
                    else:
                        pass

                    # Save the text in the dictionary
                    text_extracted[subdir][file] = text

        # space for better readability
        print("")

        # ExportJSON
        func.export_text_to_json(
        text_extracted,
        output_json_dir,
        output_file=output_file_name
        )

    else:
        # Importation du fichier JSON
        text_extracted = func.import_json_to_text(
            output_json_dir,
            input_file=output_file_name
        )
        print(
            "Le fichier JSON d'extraction de texte existe déjà, il est importé"
        )

    ##################### COPY FILES WITH MAPPING #####################
    # Print the step
    func.print_step(2, "Copie des fichiers avec mapping")

    # Convert date_mesure to datetime object
    date_obj = datetime.datetime.strptime(date_mesure, "%d/%m/%Y")

    # Copy files with mapping
    key_info_file, mapping_file = func.copy_files_with_mapping(
        text_extracted,
        pattern,
        data_dir,
        output_folder_dir,
        output_json_dir,
        client_acronym
    )

    # Importation JSON file
    key_info_dict = func.import_json_to_text(
        output_json_dir,
        input_file=key_info_file
    )

    ##################### GROUP DATA BY CHIMNEY NAME #####################
    # Print the step
    func.print_step(3, "Groupement des données par nom de cheminée")

    # group the data by chimney name
    data_per_chimney = func.group_by_chimney_name(key_info_dict, pattern)

    # save the data to JSON file
    func.save_to_json(
        data_per_chimney, output_json_dir, "data_per_chimney.json"
    )

    #####################  GET CLIENT NAME #############################
    # Print the step
    func.print_step(4, "Récupération du nom du client")

    # Get client name
    client_names = func.get_client_name_counts(key_info_dict)
    client_name = max(client_names, key=client_names.get)
    print(f"\nClient name: {client_name}")

    ##################### WORD REPORT #################################
    output_word_report_path = None
    if GENERATE_WORD_REPORT:
        # Print the step
        func.print_step(5, "Génération du rapport Word")

        # Generate word report
        output_word_report_path = func.generate_word_report(
            data_per_chimney,
            data_dir,
            output_dir,
            temp_dir,
            client_name,
            files_by_subdir,
            date_mesure,
            logo_path=os.path.join(logo_dir, "logo_rpur.png"),
        )

    ##################### DATABASE IMPLEMENTATION ##########################
    if INSERT_IN_DATABASE:
        # Print the step
        func.print_step(6, "Insertion des données dans la base de données")

        # Create the database and tables
        bdd_path = func.create_database_and_tables(bdd_dir)

        # Insert data into the database
        func.insert_client_into_db(bdd_path, client_acronym, client_name)
        func.insert_cheminee_into_db(
            bdd_path, client_acronym, data_per_chimney
        )
        func.insert_mesure_into_db(
                bdd_path,
                client_acronym,
                data_per_chimney,
                date_mesure
            )

##################### BACKUP ################################
    # Print the step
    func.print_step(7, "Création des sauvegardes")

    # Backup the database and word report
    if IS_BACKUP_CREATED:
        func.backup_database(bdd_path, backup_dir=bdd_backup_dir)
        func.backup_word_report(
            output_word_report_path,
            word_backup_dir,
            client_acronym,
            date_mesure
        )

    ##################### FINAL CLEANUP ##########################
    # Delete temporary directory
    if CLEANUP_TEMP_FILES and os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)

    # Print end message
    print("\n" + "*" * 75)
    print("==> FIN DU SCRIPT PRINCIPAL <==")
    print("*" * 75)
    func.calculate_duration(start_time)
