"""
Module d'extraction de texte des images via OCR.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""
# import
import os
import time
from tqdm import tqdm
import src.ocr_utils as func
import src.config as config


def extract_text_from_images(
        data_dir,
        output_json_dir,
        temp_dir,
        files_by_subdir,
        folder_ignored, 
        folder_ignored_dir,
        start_time
    ):
    """
    Extract text from images using OCR. If the output JSON file already exists,
    it imports the text from the file instead of re-extracting it.
    """
    # Print step
    func.print_step(1, "Extraction du texte des images")

    # Define the output file path
    output_file_name = "text_extracted.json"
    text_extracted_path = os.path.join(output_json_dir, output_file_name)

    # Check if the output file exists, if not create it
    if not os.path.exists(text_extracted_path):
        text_extracted = {}
        image_count = 0

        # Extract text from each subdirectory
        for subdir, files in tqdm(
            files_by_subdir.items(), desc="\nAnalyse des dossiers"
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
                        scale_percent=config.best_params["scale_percent"]
                    )

                    # Extract text
                    text, extract_duration = func.extract_text_easyocr(
                        image_resized_path,
                        batch_size=config.best_params["batch_size"],
                        decoder=config.best_params["decoder"],
                        adjust_contrast=config.best_params["adjust_contrast"],
                        worker=config.best_params["worker"],
                        gpu_state=config.USE_GPU_FOR_OCR
                    )

                    # Clean the french text if needed
                    if config.IS_CORRECT_TEXT_FRENCH:
                        text, clean_duration = func.correct_text_french(text)

                    # Save the text in the dictionary
                    text_extracted[subdir][file] = text

                    # Increment the image counter
                    image_count += 1

        # print number of processed images and mean duration
        print(f"\nNombre total d'images traitées : {image_count}")
        if image_count > 0:
            duration = time.time() - start_time
            mean_duration = duration / image_count
            print(f"Durée moyenne / image : {mean_duration:.1f} seconde(s)\n")

        # Export JSON
        func.export_text_to_json(
            text_extracted,
            output_json_dir,
            output_file=output_file_name
        )
    else:
        # Importation of existing JSON file if it exists
        text_extracted = func.import_json_to_text(
            output_json_dir,
            input_file=output_file_name
        )
        print(
            "Le fichier JSON d'extraction de texte existe déjà, il est importé"
        )
    
    return text_extracted
