"""
Module de génération de rapport Word pour les inspections de cheminées.

Auteurs :
Pierrick BERTHE
mail : pierrick.berthe@gmx.fr
Février 2026
"""
# imports
import os
import src.ocr_utils as func


def generate_word_report(
    data_per_chimney,
    data_dir, output_dir,
    temp_dir,
    client_name,
    files_by_subdir,
    date_mesure,
    logo_dir
):
    """
    Generate a Word report for chimney inspections.
    """
    # Initialisation output path
    output_word_report_path = None
    
    # Generation of name of output file and generation of report
    func.print_step(5, "Génération du rapport Word")
    output_file_name = f"Annexes photo 3CEP-{client_name}.docx"
    output_word_report_path = func.generate_word_report(
        data_per_chimney,
        data_dir,
        output_dir,
        temp_dir,
        client_name,
        files_by_subdir,
        date_mesure,
        logo_path=os.path.join(logo_dir, "logo_rpur.png"),
        output_file_name=output_file_name
    )

    return output_word_report_path
