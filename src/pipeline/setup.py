"""
Module de configuration pour l'initialisation des répertoires,
l'affichage des bannières, la collecte des entrées utilisateur et des fichiers.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""
# imports
import os
import sys
import importlib.metadata
import datetime
import PIL
import src.ocr_utils as func
import src.config as config


def setup_directories(project_dir):
    """
    Configure all necessary directories for the project.
    Creates them if they do not exist.
    """
    # Define project directories (inside project_dir)
    data_dir = os.path.join(project_dir, "data", "input", "source")
    output_dir = os.path.join(project_dir, "data", "output")
    output_json_dir = os.path.join(output_dir, "json")
    output_folder_dir = os.path.join(output_dir, "folder_output")
    output_log_dir = os.path.join(output_dir, "log")
    temp_dir = os.path.join(output_dir, "temp")
    logo_dir = os.path.join(project_dir, "image")
    bdd_dir = os.path.join(project_dir, "bdd")

    # Backup directories (outside project_dir)
    backup_dir = os.path.join(os.path.dirname(project_dir), "backup")
    bdd_backup_dir = os.path.join(backup_dir, "bdd")
    word_backup_dir = os.path.join(backup_dir, "word")
    json_backup_dir = os.path.join(backup_dir, "json")

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

    return {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "output_json_dir": output_json_dir,
        "output_folder_dir": output_folder_dir,
        "output_log_dir": output_log_dir,
        "temp_dir": temp_dir,
        "logo_dir": logo_dir,
        "bdd_dir": bdd_dir,
        "backup_dir": backup_dir,
        "bdd_backup_dir": bdd_backup_dir,
        "word_backup_dir": word_backup_dir,
        "json_backup_dir": json_backup_dir
    }


def print_banner_and_versions():
    """
    Print the banner and versions of Python and libraries used.
    """
    # Print banner
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
    """)

    # print the git version
    git_version = func.get_git_version()
    print(func.format_git_version(git_version))

    # Print all flags
    print("\nflags :")
    print("IS_CORRECT_TEXT_FRENCH :", config.IS_CORRECT_TEXT_FRENCH)
    print("USE_GPU_FOR_OCR        :", config.USE_GPU_FOR_OCR)
    print("GENERATE_WORD_REPORT   :", config.GENERATE_WORD_REPORT)
    print("INSERT_IN_DATABASE     :", config.INSERT_IN_DATABASE)
    print("CLEANUP_TEMP_FILES     :", config.CLEANUP_TEMP_FILES)
    print("LOG_TO_FILE            :", config.LOG_TO_FILE)
    print("IS_BACKUP_CREATED      :", config.IS_BACKUP_CREATED)

    # Print Python version and library versions
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
    print("OpenCV        : " + func.cv2.__version__)
    print("TQDM          : " + importlib.metadata.version("tqdm"))

    # Print time
    now = datetime.datetime.now().isoformat()
    print("\nCode lancé le : " + now + "\n")


def get_user_inputs(data_dir):
    """
    Ask the user for inputs such as client acronym, date of measurement,
    and folder to ignore. Returns these inputs.
    """
    # create acronym for the client name
    client_acronym = input("Entrez les initiales du nom du client : ")

    # Define the date of the measurement (format: JJ/MM/AAAA and before today)
    while True:
        try:
            date_mesure = input(
                "Entrez la date de la mesure (format JJ/MM/AAAA) : "
            )
            date_obj = datetime.datetime.strptime(date_mesure, "%d/%m/%Y")
            if date_obj > datetime.datetime.now():
                print("La date ne peut pas être dans le futur.")
                continue
            break
        except ValueError:
            print(
                "Format incorrect ou date impossible. "
                "Veuillez entrer la date au format JJ/MM/AAAA (ex. 30/06/2025)."
            )

    # Specify the folder to ignore
    folder_ignored = input("Entrez le nom exact du dossier à ignorer : ")
    folder_ignored_dir = os.path.join(data_dir, folder_ignored)

    return client_acronym, date_mesure, folder_ignored, folder_ignored_dir


def get_files_by_subdir(folder_path, folder_source_name="source"):
    """
    Get all files in the subdirectories of the given folder.
    """
    # Initialize a dictionary to hold the files by subdirectory
    files_by_subdir = {}

    # FOR each subdirectory in the folder, get the filenames
    for root, dirs, filenames in os.walk(folder_path):
        if root == folder_path:
            continue
        files_by_subdir[root.split(folder_source_name + "\\")[-1]] = filenames
    return files_by_subdir
