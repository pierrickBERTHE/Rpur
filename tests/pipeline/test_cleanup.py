"""
Module de tests unitaires pour src.pipeline.cleanup.
"""
# imports standard
import os
import tempfile
import shutil

# import for testing
import unittest
from unittest.mock import patch

# import modules to test
from src.pipeline.cleanup import cleanup_temp_files, save_client_acronym


class TestCleanupModule(unittest.TestCase):
    """
    Unit tests for src.pipeline.cleanup.
    """

    def setUp(self):
        """
        Create a temporary directory for each test.
        """
        self.tmpdir = tempfile.mkdtemp()


    def tearDown(self):
        """
        Remove the temporary directory after each test.
        """
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # patch the helper constants used in cleanup_temp_files
    @patch("src.pipeline.cleanup.CLEANUP_TEMP_FILES", True)
    def test_cleanup_temp_files_removes_files_and_directory_when_enabled(self):
        """
        When the cleanup flag is True and temp dir exists, files are removed
        and dir deleted.
        """
        # create a temporary directory with files
        temp_dir = os.path.join(self.tmpdir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # create some files inside temp_dir
        f1 = os.path.join(temp_dir, "a.tmp")
        f2 = os.path.join(temp_dir, "b.tmp")
        with open(f1, "w", encoding="utf-8") as fh:
            fh.write("data1")
        with open(f2, "w", encoding="utf-8") as fh:
            fh.write("data2")

        # act
        cleanup_temp_files(temp_dir)

        # assert that temp_dir and files are removed
        self.assertFalse(os.path.exists(temp_dir))


    # patch the helper constants used in cleanup_temp_files
    @patch("src.pipeline.cleanup.CLEANUP_TEMP_FILES", False)
    def test_cleanup_temp_files_does_nothing_when_disabled(self):
        """
        When the cleanup flag is False, the temporary directory and files
        remain untouched.
        """
        # create a temporary directory with files
        temp_dir = os.path.join(self.tmpdir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # create some files inside temp_dir
        f1 = os.path.join(temp_dir, "a.tmp")
        with open(f1, "w", encoding="utf-8") as fh:
            fh.write("data")

        # act
        cleanup_temp_files(temp_dir)

        # assert that temp_dir and files still exist
        self.assertTrue(os.path.exists(temp_dir))
        self.assertTrue(os.path.isfile(f1))

    # patch the helper constants used in cleanup_temp_files
    @patch("src.pipeline.cleanup.CLEANUP_TEMP_FILES", True)
    def test_cleanup_temp_files_handles_missing_directory_gracefully(self):
        """
        If the temp directory does not exist, function should not raise an
        exception.
        """
        # define a non-existent temp directory
        temp_dir = os.path.join(self.tmpdir, "nonexistent_temp")

        # ensure it does not exist
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # act & assert that no exception is raised
        cleanup_temp_files(temp_dir)


    def test_save_client_acronym_writes_file_with_content(self):
        """
        save_client_acronym should create client_acronym.txt with the provided
        acronym.
        """
        # prepare output log directory inside the temporary directory
        output_log_dir = os.path.join(self.tmpdir, "logs")
        os.makedirs(output_log_dir, exist_ok=True)

        # act
        acronym = "ABC"
        save_client_acronym(output_log_dir, acronym)

        # assert that the file is created with correct content
        target_file = os.path.join(output_log_dir, "client_acronym.txt")
        self.assertTrue(os.path.isfile(target_file))

        # assert content of file is correct
        with open(target_file, "r", encoding="utf-8") as fh:
            content = fh.read()
        self.assertEqual(content, acronym)


if __name__ == "__main__":
    unittest.main()
