"""
Ce fichier contient des fonctions utilitaires pour le traitement d'images et la reconnaissance de caractères.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Avril 2025
"""

import datetime
import sys
import os
import cv2
import numpy as np
import easyocr
import time
from functools import wraps
import language_tool_python
import json
from PIL import Image
import re
import shutil
import pandas as pd
import docx
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.shared import Pt
import sqlite3

# pylint: disable=no-member

class Logger(object):
    """Logger class to redirect print statements to a file.
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


def check_and_create_directories(*dirs):
    """
    Vérifie et crée tous les dossiers passés en argument s'ils n'existent pas.
    """
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)


def get_user_inputs(data_dir):
    """
    Demande à l'utilisateur les initiales du client, la date de mesure et le
    dossier à ignorer.
    """
    # create acronym for the client name
    client_acronym = input("Entrez les initiales du nom du client : ")

    # Define the date of the measurement
    while True:
        try:
            date_mesure = input(
                "Entrez la date de la mesure (format JJ/MM/AAAA) : "
                )
            date_obj = datetime.datetime.strptime(date_mesure, "%d/%m/%Y")
            break
        except ValueError:
            print("Format incorrect. Veuillez entrer la date au format JJ/MM/AAAA (ex. 30/06/2025).")

    # Specify the folder to ignore
    folder_ignored = input("Entrez le nom exact du dossier à ignorer : ")
    folder_ignored_dir = os.path.join(data_dir, folder_ignored)

    # Print the inputs
    print("\nInputs:")
    print("client_acronym     : " + client_acronym)
    print("date_mesure        : " + date_mesure)
    print("folder_ignored_dir : " + folder_ignored_dir)

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


def preprocess_black_text(image_path, output_path):
    """
    Preprocess the image to make all non-black pixels white, highlighting
    black text. Uses PIL for image loading.
    """
    try:
        # Load the image with PIL and convert to RGB then to numpy array
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = np.array(img)
    except Exception as e:
        print(f"Erreur : Impossible de charger l'image {image_path} : {e}")
        return None

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define the lower and upper bounds for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([0, 0, 0]) 

    # Create a mask for black pixels
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Apply the mask to the image
    result_image = cv2.bitwise_and(image, image, mask=mask)

    # Replace non-black pixels with white
    result_image[np.where(mask == 0)] = [255, 255, 255]

    # Save the preprocessed image
    cv2.imwrite(output_path, result_image)

    return result_image


def resize_image(input_path, output_path, scale_percent=10):
    """
    Resize an image to a given percentage of its original size.
    """
    # load the image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {input_path}")
        return None

    # Calculate the new dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize the image
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # Save the resized image
    cv2.imwrite(output_path, resized_image)

    return resized_image


def measure_time(func):
    """
    Decorator to measure the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        return result, duration
    return wrapper


@measure_time
def extract_text_easyocr(
    image_path,
    batch_size=1,
    decoder='greedy',
    adjust_contrast=True,
    worker=0
):
    """
    Extract text from image using EasyOCR
    """
    # List of character
    french_characters = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        "àâäéèêëîïôöùûüçÀÂÄÉÈÊËÎÏÔÖÙÛÜÇ' -"
    )

    # Execute OCR
    reader = easyocr.Reader(['fr'], gpu=False, verbose=False)
    text = reader.readtext(
        image_path,
        detail=0,
        batch_size=batch_size,
        decoder=decoder,
        adjust_contrast=adjust_contrast,
        workers=worker,
        allowlist=french_characters
    )
    return "\n".join(text)


@measure_time
def correct_text_french(text):
    """
    Correct the text in French using LanguageTool.
    """
    tool = language_tool_python.LanguageTool('fr')
    corrected = tool.correct(text)
    return corrected


def export_text_to_json(
        text_extracted,
        output_dir,
        output_file="text_extracted.json"
    ):
    """
    Export the text_extracted dictionary to a JSON file in the specified
    directory.
    """
    # Create the output path
    file_path = os.path.join(output_dir, output_file)

    try:
        # Save the dictionary to a JSON file
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(text_extracted, json_file, ensure_ascii=False, indent=4)
        print(f"Dictionnaire exporté dans le fichier : {file_path}")

    # Error handling
    except Exception as e:
        print(f"Erreur lors de l'exportation : {e}")


def clean_text(text):
    '''
    Clean the text by removing unwanted characters and formatting.
    '''

    # Replace line break (\n) and carriage return (-r) with space
    text = re.sub(r"[\n\r]", " ", text)

    # Keep only letters & digits (\w), spaces (\s), and dashes (-)
    text = re.sub(r"[^\w\s'-]", "", text)

    # Replace multiple spaces (\s+) with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove trailing spaces and capitale letter
    return text.strip().lower()


def extract_key_info(text, pattern=r"[a-zA-Z]\d+"):
    """
    Extract key information from the text.
    """
    # Search for the pattern in the text
    matches = list(re.finditer(pattern, text))

    if matches:

        # Get the start and end indices of the match
        first_start = matches[0].start()
        client_name = text[:first_start].strip()

        # Extract all chimney names and remarks
        chimney_names = []
        remarks = text[matches[-1].end():].strip()

        for match in matches:
            chimney_names.append(match.group())

        return {
            "client_name": client_name,
            "chimney_name": chimney_names,
            "remarks": remarks
        }
    else:

        # Log a warning if no match is found
        print(f"\nWarning: No chimney found in the text: {text}")

        # If no match is found, return None for chimney name and remarks
        return {
            "client_name": "",
            "chimney_name": "",
            "remarks": ""
        }


def generate_filename(
        text,
        pattern,
        client_acronym,
        output_dir,
        extension="jpg"
    ):
    """
    Generate filenames based on the extracted text.
    """
    # Clean the text
    cleaned_text = clean_text(text)

    # Extract key information from the cleaned text
    key_info = extract_key_info(cleaned_text, pattern)

    # Initialize a list to store generated filenames
    filenames = []

    # Handle the case where chimney_name is empty
    if not key_info["chimney_name"]:

        # Use the original filename if no chimney name is found
        filenames.append("Same_as_original")

    else:
        # Handle multiple chimney names
        for chimney_name in key_info["chimney_name"]:

            # Create the filename
            filename = f"{client_acronym}_{chimney_name}.{extension}"
            filepath = os.path.join(output_dir, filename)

            # Add a counter to the filename if it already exists
            counter = 1
            while os.path.exists(filepath):
                filename = f"{client_acronym}_{chimney_name}_{counter}.{extension}"
                filepath = os.path.join(output_dir, filename)
                counter += 1

            # Add the filename to the list
            filenames.append(filename)

    return filenames, key_info


def copy_files_with_mapping(
        text_extracted,
        pattern,
        input_dir,
        output_folder_dir,
        output_json_dir,
        client_acronym,
        key_info_file="key_info.json",
        mapping_file="file_mapping.json"
    ):
    """
    Copy files from the input directory to the output directory with new
    names based on extracted text.
    """
    # Initialise dictionaries for the mapping and key information
    mapping, key_info_dict = {}, {}

    # Create the output directory if it doesn't exist
    for subdir, files in text_extracted.items():
        subdir = subdir.split("\\")[-1]  # Get the last part of the path
        output_subdir = os.path.join(output_folder_dir, subdir)
        os.makedirs(output_subdir, exist_ok=True)
        mapping[subdir], key_info_dict[subdir] = {}, {}
        print(f"\nSUBDIRECTORY: {subdir}")
        print("-" * 75)

        # FOR each file, generate new filenames
        for file, text in files.items():
            new_filenames, key_info = generate_filename(
                text,
                pattern,
                client_acronym,
                output_dir=output_folder_dir
            )

            # Store the key information in the dictionary
            key_info_dict[subdir][file] = key_info

            # Paths for the old file
            old_path = os.path.join(input_dir, subdir, file)

            # Copy the file to the new location with each generated name
            copied_files = []
            for new_filename in new_filenames:
                # Use the original filename if "Same_as_original" is returned
                if new_filename == "Same_as_original":
                    new_filename = file

                # Generate a new path for the copied file
                new_path = os.path.join(output_subdir, new_filename)
                shutil.copy(old_path, new_path)
                copied_files.append(new_filename)

                # Print each copied file on a separate line
                print(f"Copied: {file} -> {new_filename}")

            # Add the mapping to the dictionary
            mapping[subdir][file] = copied_files

    # Save key information and mapping to JSON files
    print("\nExports des fichiers JSON : ")
    export_text_to_json(key_info_dict, output_json_dir, key_info_file)
    export_text_to_json(mapping, output_json_dir, mapping_file)

    return key_info_file, mapping_file


def import_json_to_text(input_dir, input_file="text_extracted.json"):
    """
    Import the JSON file in the specified directory.
    """
    # Create the input path
    file_path = os.path.join(input_dir, input_file)

    try:
        # Load the JSON file to dicctionnary
        with open(file_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        print(f"Dictionnaire importé depuis le fichier : {file_path}")
        return data

    # Error handling
    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' n'existe pas.")
    except json.JSONDecodeError:
        print(f"Erreur : Le fichier '{file_path}' n'est pas un fichier JSON valide.")
    except Exception as e:
        print(f"Erreur inattendue lors de l'importation du fichier JSON : {e}")


def sort_key(chimney_name, pattern):
    """
    Extract all alphabetical and numerical parts of the chimney name
    for sorting.
    """
    # Find all matches of the pattern in the chimney name
    matches = re.findall(pattern, chimney_name)
    if matches:
        # If match is a tuple (from groups), flatten if needed
        if isinstance(matches[0], tuple):
            sorted_parts = [(m[0], int(m[1])) for m in matches]
        else:
            # fallback: treat as string, not tuple
            sorted_parts = [(matches[0], 0)]
        return sorted_parts
    return [(chimney_name, 0)]


def group_by_chimney_name(data, pattern=r"[a-zA-Z]\d+"):
    """
    Group the data by chimney name and return a dictionary sorted by chimney
    name in alphabetical order and numerical value. 
    """
    # Initialize an empty dictionary to hold the grouped data
    grouped_data = {}
    counter = 1

    # Iterate through the data and group by chimney name
    for subdir, files in data.items():
        for file, info in files.items():
            chimney_names = info["chimney_name"]

            # Ensure chimney_names is a list
            if not isinstance(chimney_names, list):
                chimney_names = [chimney_names]

            # Handle the case where chimney_names is empty
            if not chimney_names or chimney_names == [""]:
                chimney_names = [f"No_chimney_{counter}"]
                counter += 1

            # Iterate through each chimney name and group the data
            for chimney_name in chimney_names:
                if chimney_name not in grouped_data:
                    grouped_data[chimney_name] = {}
                grouped_data[chimney_name][subdir] = {
                    "file": file,
                    "client_name": info["client_name"],
                    "remarks": info["remarks"]
                }

    # Sort the dictionary by the custom sort key
    grouped_data = dict(sorted(
        grouped_data.items(),
        key=lambda item: sort_key(item[0], pattern)
    ))

    return grouped_data


def save_to_json(data, output_dir, filename):
    """
    Save data to a JSON file in the specified output directory.
    """
    # create the file path
    file_path = os.path.join(output_dir, filename)

    # save the data to JSON file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\nDonnées sauvegardées dans le fichier JSON : {file_path}")
        return file_path

    # Error handling
    except FileNotFoundError:
        print(f"Erreur : Le répertoire de sortie '{output_dir}' n'existe pas.")
    except PermissionError:
        print(f"Erreur : Permission refusée pour écrire dans le répertoire '{output_dir}'.")
    except Exception as e:
        print(f"Erreur inattendue lors de la sauvegarde du fichier JSON : {e}")
    finally:
        # Close the file if it was opened
        if 'f' in locals():
            f.close()


def get_client_name_counts(data):
    """
    Retrieves the count of occurrences for each client name using Pandas.
    """
    # Flatten the data into a list of client names
    client_names = [
        info["client_name"]
        for subdir, files in data.items()
        for file, info in files.items()
    ]

    # Create a Pandas Series and count occurrences
    client_name_series = pd.Series(client_names)
    client_name_counts = client_name_series.value_counts()

    return client_name_counts.to_dict()


def compress_image(
        image_path,
        temp_dir,
        max_width=800,
        max_height=800,
        quality=50
    ):
    """
    Reduce picture size using dimension et quality reduction
    """
    # Load picture with PIL
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image = np.array(img)
    except Exception as e:
        print(f"Erreur : Impossible de charger l'image {image_path} : {e}")
        return image_path

    # Obtain picture dimension
    height, width = image.shape[:2]

    # Calculate dimension factor
    scale = min(max_width / width, max_height / height, 1.0)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize picture
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Temporary file path fo compressed picture
    compressed_image_path = os.path.join(temp_dir, "temp_compressed_image.jpg")

    # Save compressed picture with reducted quality
    cv2.imwrite(compressed_image_path, resized_image, [cv2.IMWRITE_JPEG_QUALITY, quality])

    return compressed_image_path


def add_page_number_field(paragraph):
    """
    Add a page number field to a Word document paragraph.
    """
    run = paragraph.add_run("Page ")

    # Add PAGE field (current page number)
    fldChar_begin = docx.oxml.OxmlElement('w:fldChar')
    fldChar_begin.set(docx.oxml.ns.qn('w:fldCharType'), 'begin')
    instrText = docx.oxml.OxmlElement('w:instrText')
    instrText.text = "PAGE"
    fldChar_end = docx.oxml.OxmlElement('w:fldChar')
    fldChar_end.set(docx.oxml.ns.qn('w:fldCharType'), 'end')
    run._r.append(fldChar_begin)
    run._r.append(instrText)
    run._r.append(fldChar_end)
    run.add_text(" / ")

    # Add NUMPAGES field (total number of pages)
    fldChar_begin2 = docx.oxml.OxmlElement('w:fldChar')
    fldChar_begin2.set(docx.oxml.ns.qn('w:fldCharType'), 'begin')
    instrText2 = docx.oxml.OxmlElement('w:instrText')
    instrText2.text = "NUMPAGES"
    fldChar_end2 = docx.oxml.OxmlElement('w:fldChar')
    fldChar_end2.set(docx.oxml.ns.qn('w:fldCharType'), 'end')
    run._r.append(fldChar_begin2)
    run._r.append(instrText2)
    run._r.append(fldChar_end2)


def generate_word_report(
        data_per_chimney,
        input_dir,
        output_dir,
        temp_dir,
        client_name,
        files_by_subdir,
        logo_path,
        output_file_name="rapport_extraction.docx"
    ):
    """
    Generate a Word report containing compressed images, extracted information,
    files_by_subdir keys, and a logo in the header.
    """
    # Create a Word document
    document = Document()

    # Add a header with the logo
    section = document.sections[0]
    header = section.header
    header_paragraph = header.paragraphs[0]
    header_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT

    # Add the logo to the header
    if os.path.exists(logo_path):
        run = header_paragraph.add_run()
        run.add_picture(logo_path, width=Inches(1.0))

    # Add the main centered title
    title = document.add_paragraph()
    title.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = title.add_run(
        "Annexe photo rapport diagnostic en entretien 3CEP - 2025"
    )
    run.bold = True
    run.font.size = Pt(16)

    # Add the centered subtitle
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    run = subtitle.add_run(client_name.capitalize())
    run.bold = True
    run.font.size = Pt(14)

    # === ADD PAGE NUMBER AT THE BOTTOM ===
    footer = section.footer
    footer_paragraph = footer.paragraphs[0]
    footer_paragraph.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    add_page_number_field(footer_paragraph)

    # Loop through chimneys and their information
    for conduit, subdirs in data_per_chimney.items():

        # Skip if the chimney name is "No_chimney"
        if "No_chimney" in conduit:
            continue

        # Add the chimney name
        title_1 = document.add_paragraph()
        title_1.alignment = WD_PARAGRAPH_ALIGNMENT.LEFT
        run = title_1.add_run(f"\nConduit: {conduit}")
        run.bold = True
        run.underline = True
        run.font.size = Pt(14)        

        # Create a table with 3 columns: Location, Photo, Remark
        n_rows = len(files_by_subdir.keys())
        table = document.add_table(rows=n_rows, cols=3)
        table.style = 'Table Grid'
        table.autofit = False
        table.columns[0].width = Inches(2.0)  # Width for location
        table.columns[1].width = Inches(2.5)  # Width for photo
        table.columns[2].width = Inches(3.0)  # Width for remarks

        # Fill the first column with the names from files_by_subdir
        keys_list = list(files_by_subdir.keys())
        for i, key in enumerate(keys_list):
            split_key = key.split("\\")[-1]
            table.cell(i, 0).text = split_key

        # # Only split if key contains '\\'
        for i, key in enumerate(keys_list):
            split_key = key.split("\\")[-1] if "\\" in key else key

            # Check if the split key exists in subdirs
            if split_key in subdirs:
                info = subdirs[split_key]
                file = info["file"]
                remarks = info["remarks"]

                # Photo column
                image_path = os.path.join(input_dir, key, file)
                if os.path.exists(image_path):
                    compressed_image_path = compress_image(image_path, temp_dir)
                    cell = table.cell(i, 1)
                    paragraph = cell.paragraphs[0]
                    run = paragraph.add_run()
                    run.add_picture(compressed_image_path, width=Inches(1.5))
                    paragraph.paragraph_format.space_before = Pt(4)
                    paragraph.paragraph_format.space_after = Pt(4)

                # Remark column
                table.cell(i, 2).text = remarks if remarks else ""
            else:
                table.cell(i, 1).text = ""
                table.cell(i, 2).text = ""

    # Build the full path for the output file
    output_file_path = os.path.join(output_dir, output_file_name)

    # Save the Word document
    document.save(output_file_path)
    print(f"\nRapport généré: {output_file_path}")


def create_database_and_tables(bdd_dir, db_name="bdd_airpur.db"):
    """
    Création d'une base de données SQLite et de ses tables.
    Si la base de données existe déjà, elle n'est pas recréée.
    """
    # Build the database file path
    bdd_path = os.path.join(bdd_dir, db_name)

    # If the database file does not exist, create it and the tables
    if not os.path.exists(bdd_path):

        # Connect to the SQLite database
        conn = sqlite3.connect(bdd_path, timeout=10)
        cur = conn.cursor()

        # Create the 'clients' table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS clients (
            client_id TEXT PRIMARY KEY,
            nom TEXT
        )
        """)

        # Create the 'cheminees' table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cheminees (
            client_id TEXT,
            cheminee_id TEXT,
            localisation TEXT,
            remarques TEXT,
            PRIMARY KEY (cheminee_id, client_id, localisation),
            FOREIGN KEY (client_id) REFERENCES clients(client_id)
        )
        """)

        # Create the 'mesures' table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS mesures (
            client_id TEXT,
            cheminee_id INTEGER,
            mesure_id INTEGER,
            date_mesure DATETIME,
            PRIMARY KEY (mesure_id, client_id, cheminee_id),
            FOREIGN KEY (cheminee_id) REFERENCES cheminees(cheminee_id),
            FOREIGN KEY (client_id) REFERENCES clients(client_id)
        )
        """)

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        print(f"\nBase de données créée : {bdd_path}\n")
    else:
        print(f"\nBase de données déjà existante : {bdd_path}\n")

    return bdd_path


def insert_client_into_db(bdd_path, client_acronym, client_name):
    """
    Insère un client dans la base de données SQLite.
    Si le client existe déjà, il ne sera pas inséré à nouveau.
    """
    print("\nTABLE clients : ")
    try:
        # Connect to the database
        conn = sqlite3.connect(bdd_path, timeout=10)
        cur = conn.cursor()

        # Insert client data into the table
        try:
            cur.execute(
                "INSERT INTO clients (client_id, nom) VALUES (?, ?)",
                (client_acronym, client_name)
            )
            conn.commit()
            print(f"\nClient '{client_name}' inséré avec succès dans la base de données : {bdd_path}")

        # If the client already exists, print a message
        except sqlite3.IntegrityError:
            print(f"Le client '{client_name}' existe déjà dans la base de données : {bdd_path}")

        # Print error if insertion fails
        except Exception as e:
            print(f"Erreur lors de l'insertion du client : {e}")

    # Ensure the connection is closed
    finally:
        try:
            conn.close()
        except Exception as e:
            print(f"Erreur lors de la fermeture de la connexion : {e}")


def insert_cheminee_into_db(bdd_path, client_acronym, data_per_chimney):
    """
    Insertion des données des cheminées dans la base de données SQLite.
    """
    print("\nTABLE cheminées : ")
    try:
        # Connect to the database
        conn = sqlite3.connect(bdd_path, timeout=10)
        cur = conn.cursor()

        # FOR each chimney in the data_per_chimney dictionary
        for cheminee_id, subdirs in data_per_chimney.items():

            # Skip if the chimney name is "No_chimney"
            if "No_chimney" in cheminee_id:
                continue

            # Insert each subdirectory as a chimney using localisation
            for subdir, info in subdirs.items():

                # Keep the longest word in the subdirectory name
                words = subdir.split()
                max_length_word = max(words, key=len) if words else ""

                # Keep the remarks if they exist
                remarques = info.get("remarks", "")

                # Insert the chimney data into the table
                try:
                    cur.execute(
                        "INSERT INTO cheminees (client_id, cheminee_id, localisation, remarques) VALUES (?, ?, ?, ?)",
                        (client_acronym, cheminee_id, max_length_word, remarques)
                    )

                # Exception handling for duplicate entries
                except sqlite3.IntegrityError:
                    print(f"La cheminée '{cheminee_id}' existe déjà dans la base de données : {bdd_path}")
                except Exception as e:
                    print(f"Erreur lors de l'insertion de la cheminée {cheminee_id} : {e}")

        conn.commit()

    # Ensure the connection is closed
    except Exception as e:
        print(f"Erreur de connexion ou d'exécution SQL : {e}")

    # Ensure the connection is closed
    finally:
        try:
            conn.close()
        except Exception as e:
            print(f"Erreur lors de la fermeture de la connexion : {e}")


def insert_mesure_into_db(
        bdd_path,
        client_acronym,
        data_per_chimney,
        date_mesure
    ):
    """
    Insertion des mesures dans la base de données SQLite.
    """
    print("\nTABLE mesures : ")
    try:
        # Connect to the database
        conn = sqlite3.connect(bdd_path, timeout=3)
        cur = conn.cursor()

        # FOR each chimney in the data_per_chimney dictionary
        for cheminee_id, subdirs in data_per_chimney.items():

            # Skip if the chimney name is "No_chimney"
            if "No_chimney" in cheminee_id:
                continue

            # FOR each subdirectory in the chimney
            for subdir, info in subdirs.items():

                # Format the measurement date to SQL format
                date_mesure_str = date_mesure
                try:
                    date_obj = datetime.datetime.strptime(
                        date_mesure_str, "%d/%m/%Y"
                    )
                    date_mesure_sql = date_obj.strftime("%Y-%m-%d")
                except Exception:
                    date_mesure_sql = date_mesure_str

                # Find the last mesure_id and date for this client and chimney
                cur.execute("""
                    SELECT MAX(mesure_id), MAX(date_mesure) FROM mesures
                    WHERE client_id = ? AND cheminee_id = ?
                """, (client_acronym, cheminee_id))
                row = cur.fetchone()
                last_mesure_id, last_date_in_db = row if row else (None, None)

                # If no previous measure, start at 1
                if last_mesure_id is None:
                    mesure_id = 1
                else:
                    # If the current date is newer, increment mesure_id
                    if last_date_in_db is not None and date_mesure_sql > last_date_in_db:
                        mesure_id = last_mesure_id + 1
                    else:
                        mesure_id = last_mesure_id

                # Check if the measure already exists
                cur.execute("""
                    SELECT COUNT(*) FROM mesures
                    WHERE client_id = ? AND cheminee_id = ? AND mesure_id = ?
                """, (client_acronym, cheminee_id, mesure_id))
                exists = cur.fetchone()[0]

                if exists:
                    print(f"Déjà présent (clé primaire) : (client_id={client_acronym}, cheminee_id={cheminee_id}, mesure_id={mesure_id})")
                    continue

                try:
                    cur.execute(
                        "INSERT INTO mesures (client_id, cheminee_id, mesure_id, date_mesure) VALUES (?, ?, ?, ?)",
                        (client_acronym, cheminee_id, mesure_id, date_mesure_sql)
                    )
                    print(f"Mesure insérée pour {cheminee_id} avec mesure_id={mesure_id} et date={date_mesure_sql}")

                # Exception handling for duplicate entries
                except Exception as e:
                    print(f"Erreur lors de l'insertion de la mesure pour {cheminee_id} : {e}")
        conn.commit()

    # Error handling for connection issues
    except Exception as e:
        print(f"Erreur de connexion ou d'exécution SQL : {e}")

    # Ensure the connection is closed
    finally:
        try:
            conn.close()
        except Exception as e:
            print(f"Erreur lors de la fermeture de la connexion : {e}")


def print_step(step_num, message):
    """
    Print a formatted step message for process reporting.
    """
    print("\n\n" + "*" * 75)
    print(f"==> STEP {step_num} : {message} <==")
    print("*" * 75)
