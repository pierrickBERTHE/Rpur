"""
Module de test unitaire pour le module src.pipeline.processing.
"""
# imports standard
from io import StringIO
import sys

# imports for testing
import unittest
from unittest.mock import patch

# import the function to be tested
import src.pipeline.processing as processing
from src.pipeline.processing import process_files_and_data, get_client_name


class TestProcessingModule(unittest.TestCase):
    """
    Unit tests for src.pipeline.processing.
    """

    # patch the helper functions used in process_files_and_data
    @patch("src.pipeline.processing.func.save_to_json")
    @patch("src.pipeline.processing.func.group_by_chimney_name")
    @patch("src.pipeline.processing.func.import_json_to_text")
    @patch("src.pipeline.processing.func.copy_files_with_mapping")
    @patch("src.pipeline.processing.func.print_step")
    def test_process_files_and_data_calls_required_helpers(
        self,
        mock_print_step,
        mock_copy_files_with_mapping,
        mock_import_json_to_text,
        mock_group_by_chimney_name,
        mock_save_to_json
    ):
        """
        Ensure process_files_and_data orchestrates copy, import, grouping and
        saving.
        """
        # Arrange test data
        text_extracted = {"img1.jpg": "text1"}
        data_dir = "/data"
        output_folder_dir = "/out/folder"
        output_json_dir = "/out/json"
        client_acronym = "ABC"

        # Configure mocks
        mock_copy_files_with_mapping.return_value = (
            "key_info.json", "mapping.json"
        )
        mock_import_json_to_text.return_value = {
            "entry1": {"client_name": "ClientA"}
            }
        mock_group_by_chimney_name.return_value = {"chimney1": []}

        # Act
        key_info_dict, data_per_chimney = process_files_and_data(
            text_extracted,
            data_dir,
            output_folder_dir,
            output_json_dir,
            client_acronym
        )

        # Assert return values
        self.assertEqual(key_info_dict, {"entry1": {"client_name": "ClientA"}})
        self.assertEqual(data_per_chimney, {"chimney1": []})

        # Assert helper calls
        mock_print_step.assert_any_call(2, "Copie des fichiers avec mapping")
        mock_copy_files_with_mapping.assert_called_once()
    
        # verify first arg of copy is the text_extracted mapping
        copy_args = mock_copy_files_with_mapping.call_args[0]
        self.assertIs(copy_args[0], text_extracted)
    
        # pattern is imported at module level
        self.assertEqual(copy_args[1], processing.pattern)
        self.assertEqual(copy_args[2], data_dir)
        self.assertEqual(copy_args[3], output_folder_dir)
        self.assertEqual(copy_args[4], output_json_dir)
        self.assertEqual(copy_args[5], client_acronym)

        # Assert helper functions been called with expected arguments
        mock_import_json_to_text.assert_called_once_with(
            output_json_dir, input_file="key_info.json"
            )
        mock_print_step.assert_any_call(
            3, "Groupement des données par nom de cheminée"
            )
        mock_group_by_chimney_name.assert_called_once_with(
            key_info_dict, processing.pattern
        )
        mock_save_to_json.assert_called_once_with(
            data_per_chimney, output_json_dir, "data_per_chimney.json"
        )


    # patch the helper functions used in get_client_name
    @patch("src.pipeline.processing.func.print_step")
    @patch("src.pipeline.processing.func.get_client_name_counts")
    def test_get_client_name_selects_most_frequent_and_prints(
        self,
        mock_get_client_name_counts,
        mock_print_step
    ):
        """
        Ensure get_client_name returns the most frequent client and prints it.
        """
        # Arrange
        key_info_dict = {
            "a": {"client_name": "ClientA"},
            "b": {"client_name": "ClientB"}
        }
        # return counts where ClientA is most frequent
        mock_get_client_name_counts.return_value = {"ClientA": 5, "ClientB": 2}

        # capture stdout
        captured = StringIO()
        original_stdout = sys.stdout
        try:
            sys.stdout = captured
            result = get_client_name(key_info_dict)
        finally:
            sys.stdout = original_stdout

        # Assert
        self.assertEqual(result, "ClientA")
        output = captured.getvalue()
        self.assertIn("Client name: ClientA", output)
        mock_print_step.assert_called_once_with(
            4, "Récupération du nom du client"
        )


if __name__ == "__main__":
    unittest.main()
