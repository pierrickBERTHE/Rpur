"""
Module de test unitaire pour le module src.pipeline.setup.
"""
# imports standard
import os
import sys
import tempfile
import shutil

# Imports for testing
import unittest
from io import StringIO
from unittest.mock import patch, Mock

# import modules to test
from src.pipeline.setup import (
    setup_directories,
    print_banner_and_versions,
    get_user_inputs,
    get_files_by_subdir
)


class TestSetupModule(unittest.TestCase):
    """
    Unit tests for src.pipeline.setup.
    """

    def setUp(self):
        """
        Prepare a temporary project directory for tests that need files.
        """
        self.tmpdir = tempfile.mkdtemp()
        self.project_dir = os.path.join(self.tmpdir, "project")
        os.makedirs(self.project_dir, exist_ok=True)


    def tearDown(self):
        """
        Clean up temporary directory after each test.
        """
        shutil.rmtree(self.tmpdir)


    # patch the helper functions used in setup_directories
    @patch("src.pipeline.setup.func.check_and_create_directories")
    def test_setup_directories_calls_check_and_create_directories(
        self, mock_check
    ):
        """
        Ensure setup_directories computes paths and calls helper to create them.
        """
        # Act
        result = setup_directories(self.project_dir)

        # Assert returned mapping contains expected keys
        expected_keys = {
            "data_dir",
            "output_dir",
            "output_json_dir",
            "output_folder_dir",
            "output_log_dir",
            "temp_dir",
            "logo_dir",
            "bdd_dir",
            "backup_dir",
            "bdd_backup_dir",
            "word_backup_dir",
            "json_backup_dir",
        }
        self.assertTrue(expected_keys.issubset(set(result.keys())))

        # Assert helper was called
        mock_check.assert_called_once()


    # patch the helper functions used in print_banner_and_versions
    @patch("src.pipeline.setup.func.get_git_version", return_value="gitrev")
    @patch(
        "src.pipeline.setup.func.format_git_version",
        return_value="formatted_gitrev"
    )
    @patch("src.pipeline.setup.importlib.metadata.version", return_value="1.0")
    @patch("src.pipeline.setup.func.docx", new=Mock(__version__="0.1"))
    @patch("src.pipeline.setup.func.easyocr", new=Mock(__version__="2.0"))
    @patch("src.pipeline.setup.func.np", new=Mock(__version__="1.21"))
    @patch("src.pipeline.setup.func.pd", new=Mock(__version__="1.3"))
    @patch("src.pipeline.setup.func.cv2", new=Mock(__version__="4.5"))
    @patch("src.pipeline.setup.PIL", new=Mock(__version__="9.0"))
    def test_print_banner_and_versions_outputs_expected_lines(
        self,
        mock_meta_version,
        mock_format_git,
        mock_get_git
    ):
        """
        Capture stdout and assert banner and formatted git version are printed.
        """
        captured = StringIO()
        original_stdout = sys.stdout
        try:
            sys.stdout = captured
            print_banner_and_versions()
        finally:
            sys.stdout = original_stdout
        output = captured.getvalue()
        self.assertIn("RRRRR", output)
        self.assertIn("formatted_gitrev", output)
        self.assertIn("flags :", output)


    # patch the helper functions used in get_user_inputs
    @patch(
            "builtins.input",
            side_effect=["ABC", "01/08/2025", "ignored_folder"]
        )
    def test_get_user_inputs_returns_expected_tuple(self, mock_input):
        """
        Simulate user input and verify returned 4-tuple and folder path
        composition.
        """
        # Arrange test data
        data_dir = os.path.join(self.project_dir, "data", "input", "source")
        os.makedirs(data_dir, exist_ok=True)

        # Act
        (
            client_acronym,
            date_mesure,
            folder_ignored,
            folder_ignored_dir
        ) = get_user_inputs(data_dir)

        # Assert
        self.assertEqual(client_acronym, "ABC")
        self.assertEqual(date_mesure, "01/08/2025")
        self.assertEqual(folder_ignored, "ignored_folder")
        self.assertEqual(folder_ignored_dir, os.path.join(
            data_dir, "ignored_folder")
        )


    def test_get_files_by_subdir_discovers_files(self):
        """
        Create a small directory tree and verify get_files_by_subdir 
        filenames grouped by subdir.
        """
        # Arrange test data - create dummy files
        base = os.path.join(self.tmpdir, "source")
        sub1 = os.path.join(base, "subdir1")
        sub2 = os.path.join(base, "subdir2")
        os.makedirs(sub1, exist_ok=True)
        os.makedirs(sub2, exist_ok=True)
        f1 = os.path.join(sub1, "a.jpg")
        f2 = os.path.join(sub1, "b.jpg")
        f3 = os.path.join(sub2, "c.jpg")
        open(f1, "w").close()
        open(f2, "w").close()
        open(f3, "w").close()

        # Act
        result = get_files_by_subdir(base)

        # Assert dummy files existing
        all_files = set()
        for v in result.values():
            all_files.update(v)
        self.assertTrue({"a.jpg", "b.jpg", "c.jpg"}.issubset(all_files))


if __name__ == "__main__":
    unittest.main()
