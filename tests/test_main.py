"""
Module de test pour la fonction main() de src/main.py.
Ce test vérifie l'exécution complète de main() avec tous les flags de
configuration activés et désactivés, en utilisant des mocks pour toutes les
fonctions dépendantes afin d'assurer une isolation totale du test.
"""
# Imports standard library
from io import StringIO

# imports for testing
import unittest
from unittest.mock import patch

# import modules to test
from src.main import main


class MockDirs(dict):
    """
    Helper class to create a mocked directory structure.
    """
    def __init__(self):
        super().__init__()
        self.update(
            {
                'data_dir': '/tmp/data',
                'output_dir': '/tmp/output',
                'output_json_dir': '/tmp/json',
                'output_folder_dir': '/tmp/folder',
                'output_log_dir': '/tmp/logs',
                'temp_dir': '/tmp/temp',
                'logo_dir': '/tmp/logos',
                'bdd_dir': '/tmp/bdd',
                'backup_dir': '/tmp/backup',
                'bdd_backup_dir': '/tmp/backup/db',
                'word_backup_dir': '/tmp/backup/word',
                'json_backup_dir': '/tmp/backup/json'
            }
        )


class TestMainFunctionExecution(unittest.TestCase):
    """
    Test suite for main() function execution and orchestration.
    """

    # patch the helper functions used in main
    @patch("src.main.func.calculate_duration")
    @patch("src.main.save_client_acronym")
    @patch("src.main.cleanup_temp_files")
    @patch("src.main.create_backups")
    @patch("src.main.insert_into_database")
    @patch("src.main.generate_word_report")
    @patch("src.main.get_client_name")
    @patch("src.main.process_files_and_data")
    @patch("src.main.extract_text_from_images")
    @patch("src.main.func.save_to_json")
    @patch("src.main.get_files_by_subdir")
    @patch("src.main.get_user_inputs")
    @patch("src.main.print_banner_and_versions")
    @patch("src.main.setup_directories")
    @patch("os.getcwd")
    @patch("src.main.func.Logger", return_value=StringIO())
    @patch("src.main.config.LOG_TO_FILE", False)
    @patch("src.main.config.GENERATE_WORD_REPORT", False)
    @patch("src.main.config.INSERT_IN_DATABASE", False)
    @patch("src.main.config.IS_BACKUP_CREATED", False)
    def test_main_execution_with_all_flags_disabled(
        self,
        mock_logger,
        mock_getcwd,
        mock_setup_dirs,
        mock_banner,
        mock_user_inputs,
        mock_get_files,
        mock_save_json,
        mock_extract,
        mock_process,
        mock_get_name,
        mock_report,
        mock_insert_db,
        mock_backup,
        mock_cleanup,
        mock_save_acronym,
        mock_calculate_duration,
    ):
        """
        Test main() execution with all optional features disabled.
        """
        # Arrange - Setup mock return values
        mock_getcwd.return_value = 'C:\\project\\src'
        mock_setup_dirs.return_value = MockDirs()
        mock_user_inputs.return_value = (
            'ABC', '2025-08-01', 'ignored_folder', '/data/ignored_folder'
        )
        mock_get_files.return_value = {'subdir1': ['img1.jpg']}
        mock_extract.return_value = {'img1.jpg': 'extracted text'}
        mock_process.return_value = ({'client': 'ABC'}, {'chimney1': []})
        mock_get_name.return_value = 'Client ABC'

        # Act
        print(f"Mock return value: {mock_user_inputs.return_value}")
        main()

        # Optional functions should NOT be called
        mock_report.assert_not_called()
        mock_insert_db.assert_not_called()
        mock_backup.assert_not_called()

        # Assert others functions been called
        mock_setup_dirs.assert_called_once()
        mock_banner.assert_called_once()
        mock_user_inputs.assert_called_once()
        mock_get_files.assert_called_once()
        mock_extract.assert_called_once()
        mock_process.assert_called_once()
        mock_get_name.assert_called_once()
        mock_cleanup.assert_called_once()
        mock_save_acronym.assert_called_once()
        mock_calculate_duration.assert_called_once()
        mock_save_json.assert_called_once()


    # patch the helper functions used in main
    @patch("src.main.func.calculate_duration")
    @patch("src.main.save_client_acronym")
    @patch("src.main.cleanup_temp_files")
    @patch("src.main.create_backups")
    @patch("src.main.insert_into_database")
    @patch("src.main.generate_word_report")
    @patch("src.main.get_client_name")
    @patch("src.main.process_files_and_data")
    @patch("src.main.extract_text_from_images")
    @patch("src.main.func.save_to_json")
    @patch("src.main.get_files_by_subdir")
    @patch("src.main.get_user_inputs")
    @patch("src.main.print_banner_and_versions")
    @patch("src.main.setup_directories")
    @patch("os.getcwd")
    @patch("src.main.func.Logger", return_value=StringIO())
    @patch("src.main.config.LOG_TO_FILE", True)
    @patch("src.main.config.GENERATE_WORD_REPORT", True)
    @patch("src.main.config.INSERT_IN_DATABASE", True)
    @patch("src.main.config.IS_BACKUP_CREATED", True)
    def test_main_execution_with_all_flags_enabled(
        self,
        mock_logger,
        mock_getcwd,
        mock_setup_dirs,
        mock_banner,
        mock_user_inputs,
        mock_get_files,
        mock_save_json,
        mock_extract,
        mock_process,
        mock_get_name,
        mock_report,
        mock_insert_db,
        mock_backup,
        mock_cleanup,
        mock_save_acronym,
        mock_calculate_duration,
    ):
        """
        Test main() execution with all optional features disabled.
        """
        # Arrange - Setup mock return values
        mock_getcwd.return_value = 'C:\\project\\src'
        mock_setup_dirs.return_value = MockDirs()
        mock_user_inputs.return_value = (
            'ABC', '2025-08-01', 'ignored_folder', '/data/ignored_folder'
        )
        mock_get_files.return_value = {'subdir1': ['img1.jpg']}
        mock_extract.return_value = {'img1.jpg': 'extracted text'}
        mock_process.return_value = ({'client': 'ABC'}, {'chimney1': []})
        mock_get_name.return_value = 'Client ABC'

        # Act
        print(f"Mock return value: {mock_user_inputs.return_value}")
        main()

        # Assert functions been called
        mock_setup_dirs.assert_called_once()
        mock_banner.assert_called_once()
        mock_user_inputs.assert_called_once()
        mock_get_files.assert_called_once()
        mock_extract.assert_called_once()
        mock_process.assert_called_once()
        mock_get_name.assert_called_once()
        mock_cleanup.assert_called_once()
        mock_save_acronym.assert_called_once()
        mock_calculate_duration.assert_called_once()
        mock_save_json.assert_called_once()

        # Optional functions should be called
        mock_report.assert_called_once()
        mock_insert_db.assert_called_once()
        mock_backup.assert_called_once()


if __name__ == '__main__':
    unittest.main()
