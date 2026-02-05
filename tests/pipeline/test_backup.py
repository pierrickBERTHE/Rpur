"""
Module de tests unitaires pour src.pipeline.backup.
"""
# import for testing
import unittest
from unittest import mock
from unittest.mock import patch

# import the module to be tested
from src.pipeline.backup import create_backups


class TestBackupModule(unittest.TestCase):
    """Unit tests for src.pipeline.backup."""

    # patch the helper functions used in create_backups
    @patch("src.pipeline.backup.func.backup_json")
    @patch("src.pipeline.backup.func.backup_word_report")
    @patch("src.pipeline.backup.func.backup_database")
    @patch("src.pipeline.backup.func.print_step")
    def test_create_backups_with_all_paths_provided(
        self,
        mock_print_step,
        mock_backup_database,
        mock_backup_word_report,
        mock_backup_json
    ):
        """
        Ensure all backup helpers are called when database and report paths
        are provided.
        """
        # Arrange backup parameters
        bdd_path = "/bdd/database.db"
        bdd_backup_dir = "/backup/bdd"
        output_word_report_path = "/output/report.docx"
        word_backup_dir = "/backup/word"
        output_json_dir = "/output/json"
        json_backup_dir = "/backup/json"
        client_acronym = "ABC"

        # Act
        create_backups(
            bdd_path,
            bdd_backup_dir,
            output_word_report_path,
            word_backup_dir,
            output_json_dir,
            json_backup_dir,
            client_acronym
        )

        # Assert that each function was called once with correct parameters
        mock_print_step.assert_called_once_with(7, "Création des sauvegardes")
        mock_backup_database.assert_called_once_with(
            bdd_path, backup_dir=bdd_backup_dir
        )
        mock_backup_word_report.assert_called_once_with(
            output_word_report_path,
            word_backup_dir,
            client_acronym
        )
        mock_backup_json.assert_called_once_with(
            output_json_dir,
            json_backup_dir,
            client_acronym
        )


    # patch the helper functions used in create_backups
    @patch("src.pipeline.backup.func.backup_json")
    @patch("src.pipeline.backup.func.backup_word_report")
    @patch("src.pipeline.backup.func.backup_database")
    @patch("src.pipeline.backup.func.print_step")
    def test_create_backups_skips_database_when_path_is_none(
        self,
        mock_print_step,
        mock_backup_database,
        mock_backup_word_report,
        mock_backup_json
    ):
        """
        When bdd_path is None, database backup should be skipped.
        """
        # Arrange backup parameters
        bdd_path = None
        output_word_report_path = "/output/report.docx"

        # Act
        create_backups(
            bdd_path,
            "/backup/bdd",
            output_word_report_path,
            "/backup/word",
            "/output/json",
            "/backup/json",
            "ABC"
        )

        # Assert database backup was NOT called
        mock_backup_database.assert_not_called()

        # Assert others functions were called
        mock_backup_word_report.assert_called_once()
        mock_backup_json.assert_called_once()


    # patch the helper functions used in create_backups
    @patch("src.pipeline.backup.func.backup_json")
    @patch("src.pipeline.backup.func.backup_word_report")
    @patch("src.pipeline.backup.func.backup_database")
    @patch("src.pipeline.backup.func.print_step")
    def test_create_backups_skips_word_report_when_path_is_none(
        self,
        mock_print_step,
        mock_backup_database,
        mock_backup_word_report,
        mock_backup_json
    ):
        """
        When output_word_report_path is None, word report backup should be
        skipped.
        """
        # Arrange backup parameters
        bdd_path = "/bdd/database.db"
        output_word_report_path = None

        # Act
        create_backups(
            bdd_path,
            "/backup/bdd",
            output_word_report_path,
            "/backup/word",
            "/output/json",
            "/backup/json",
            "ABC"
        )

        # Assert word report backup was NOT called
        mock_backup_word_report.assert_not_called()

        # Assert others functions were called
        mock_backup_database.assert_called_once()
        mock_backup_json.assert_called_once()
        mock_print_step.assert_called_once_with(7, "Création des sauvegardes")


    # patch the helper functions used in create_backups
    @patch("src.pipeline.backup.func.backup_json")
    @patch("src.pipeline.backup.func.backup_word_report")
    @patch("src.pipeline.backup.func.backup_database")
    @patch("src.pipeline.backup.func.print_step")
    def test_create_backups_json_always_called(
        self,
        mock_print_step,
        mock_backup_database,
        mock_backup_word_report,
        mock_backup_json
    ):
        """
        JSON backup must always be called regardless of other paths.
        """
        # Arrange backup parameters
        output_json_dir = "/output/json"
        json_backup_dir = "/backup/json"
        client_acronym = "XYZ"

        # Act - with all paths as None except JSON directories
        create_backups(
            None,
            "/backup/bdd",
            None,
            "/backup/word",
            output_json_dir,
            json_backup_dir,
            client_acronym
        )

        # Assert database and word report backups were NOT called
        mock_backup_database.assert_not_called()
        mock_backup_word_report.assert_not_called()

        # Assert others functions were called
        mock_backup_json.assert_called_once_with(
            output_json_dir,
            json_backup_dir,
            client_acronym
        )
        mock_print_step.assert_called_once_with(7, "Création des sauvegardes")


    # patch the helper functions used in create_backups
    @patch("src.pipeline.backup.func.backup_json")
    @patch("src.pipeline.backup.func.backup_word_report")
    @patch("src.pipeline.backup.func.backup_database")
    @patch("src.pipeline.backup.func.print_step")
    def test_create_backups_passes_correct_parameters_to_helpers(
        self,
        mock_print_step,
        mock_backup_database,
        mock_backup_word_report,
        mock_backup_json
    ):
        """
        Verify all parameters are correctly forwarded to backup helpers.
        """
        # Arrange backup parameters
        bdd_path = "/db/app.db"
        bdd_backup_dir = "/backups/databases"
        output_word_report_path = "/reports/final.docx"
        word_backup_dir = "/backups/reports"
        output_json_dir = "/data/json"
        json_backup_dir = "/backups/json"
        client_acronym = "CORP"

        # Act
        create_backups(
            bdd_path,
            bdd_backup_dir,
            output_word_report_path,
            word_backup_dir,
            output_json_dir,
            json_backup_dir,
            client_acronym
        )

        # Assert backup helpers received correct parameters
        db_call_args = mock_backup_database.call_args
        self.assertEqual(db_call_args[0][0], bdd_path)
        self.assertEqual(db_call_args[1]["backup_dir"], bdd_backup_dir)

        word_call_args = mock_backup_word_report.call_args[0]
        self.assertEqual(word_call_args[0], output_word_report_path)
        self.assertEqual(word_call_args[1], word_backup_dir)
        self.assertEqual(word_call_args[2], client_acronym)

        json_call_args = mock_backup_json.call_args[0]
        self.assertEqual(json_call_args[0], output_json_dir)
        self.assertEqual(json_call_args[1], json_backup_dir)
        self.assertEqual(json_call_args[2], client_acronym)

        print_call_args = mock_print_step.call_args
        self.assertEqual(print_call_args[0][0], 7)
        self.assertEqual(print_call_args[0][1], "Création des sauvegardes")


if __name__ == "__main__":
    unittest.main()
