"""
Module de test unitaire pour le module src.pipeline.extraction.
"""
# import
import os
import sys
from io import StringIO

# imprt for testing
import unittest
from unittest.mock import patch, Mock

# import the module to test
import src.pipeline
from src.pipeline.extraction import extract_text_from_images


class TestExtractionModule(unittest.TestCase):
    """
    Unit tests for src.pipeline.extraction.
    """

    def setUp(self):
        """
        Prepare common parameters used across tests.
        """
        self.data_dir = "/data"
        self.output_json_dir = "/out/json"
        self.temp_dir = "/tmp"
        self.start_time = 0.0


    def test_extract_text_processes_and_exports_when_no_json(self):
        """
        When output JSON is absent, images are processed and
        export_text_to_json is called.
        """
        # Mock files by subdirectory
        files_by_subdir = {"subdir1": ["img1.jpg"]}
        folder_ignored = []
        folder_ignored_dir = ""

        # Mock all functions involved in processing to return controlled values
        with patch(
            "src.pipeline.extraction.os.path.exists", return_value=False
            ), \
            patch(
                "src.pipeline.extraction.func.print_step"
            ) as mock_print_step, \
            patch(
                "src.pipeline.extraction.func.preprocess_black_text",
                return_value="proc_img"
            ) as mock_pre, \
            patch(
                "src.pipeline.extraction.func.resize_image",
                return_value="resized_img"
            ) as mock_resize, \
            patch(
                "src.pipeline.extraction.func.extract_text_easyocr",
                return_value=("raw_text", 0.1)
            ) as mock_extract_easy, \
            patch(
                "src.pipeline.extraction.func.correct_text_french",
                return_value=("clean_text", 0.05)
            ) as mock_correct, \
            patch(
                "src.pipeline.extraction.func.export_text_to_json"
            ) as mock_export, \
            patch(
                 "src.pipeline.extraction.tqdm", new=lambda x, **kw: x
                ), \
            patch(
                "src.pipeline.extraction.config.IS_CORRECT_TEXT_FRENCH",
                True
            ), \
            patch(
                "src.pipeline.extraction.config.best_params",
                new={
                    "scale_percent": 50,
                    "batch_size": 1,
                    "decoder": "greedy",
                    "adjust_contrast": False,
                    "worker": 1
                    }
                ), \
            patch("src.pipeline.extraction.config.USE_GPU_FOR_OCR", False):

            # Act
            result = extract_text_from_images(
                self.data_dir,
                self.output_json_dir,
                self.temp_dir,
                files_by_subdir,
                folder_ignored,
                folder_ignored_dir,
                self.start_time
            )

        # Assert results contain expected cleaned text
        self.assertIn("subdir1", result)
        self.assertEqual(result["subdir1"]["img1.jpg"], "clean_text")

        # Assert export was called
        mock_export.assert_called_once()


    def test_extract_text_imports_existing_json_when_present(self):
        """
        When the output JSON exists, function imports it and returns its
        content.
        """
        # Mock expected imported data
        expected = {"subdir1": {"img1.jpg": "stored_text"}}

        # Mock os.path.exists to simulate existing JSON file
        with patch("" \
        "src.pipeline.extraction.os.path.exists", return_value=True
        ), \
            patch(
                "src.pipeline.extraction.func.import_json_to_text",
                return_value=expected
            ) as mock_import, \
            patch(
                "src.pipeline.extraction.tqdm", new=lambda x, **kw: x
            ):

            # Act
            result = extract_text_from_images(
                self.data_dir,
                self.output_json_dir,
                self.temp_dir,
                {},
                [],
                "",
                self.start_time
            )

        # Assert imported data is returned
        mock_import.assert_called_once_with(
            self.output_json_dir, input_file="text_extracted.json"
        )
        self.assertEqual(result, expected)


    def test_ignored_folder_gets_empty_text_entries(self):
        """
        If a subdirectory matches the ignored folder, its files get empty text
        and no OCR is run.
        """
        # create files by subdirectory with one ignored and one normal
        files_by_subdir = {
            "ignored_subdir": ["a.jpg", "b.jpg"],
            "normal_subdir": ["c.jpg"]
        }

        # Define ignored folder names
        folder_ignored = "ignored_subdir"
        folder_ignored_dir = os.path.join(self.data_dir, "ignored_subdir")

        # Mock functions involved in processing
        with patch(
            "src.pipeline.extraction.os.path.exists", return_value=False
            ), \
            patch(
                "src.pipeline.extraction.func.preprocess_black_text"
            ) as mock_pre, \
            patch(
                "src.pipeline.extraction.func.export_text_to_json"
            ) as mock_export, \
            patch(
                "src.pipeline.extraction.tqdm", new=lambda x, **kw: x
            ), \
            patch(
                "src.pipeline.extraction.config.IS_CORRECT_TEXT_FRENCH",
                False
            ), \
            patch(
                "src.pipeline.extraction.config.best_params",
                new={
                    "scale_percent": 50,
                    "batch_size": 1,
                    "decoder": "greedy",
                    "adjust_contrast": False,
                    "worker": 1
                }
            ), \
            patch(
                "src.pipeline.extraction.config.USE_GPU_FOR_OCR", False
            ), \
            patch(
                "src.pipeline.extraction.func.resize_image",
                return_value="resized_img"
            ), \
            patch(
                "src.pipeline.extraction.func.extract_text_easyocr",
                return_value=("raw_text", 0.1)
            ):

            # Act
            result = extract_text_from_images(
                self.data_dir,
                self.output_json_dir,
                self.temp_dir,
                files_by_subdir,
                folder_ignored,
                folder_ignored_dir,
                self.start_time
            )

        # Assert ignored subdir files have empty text
        self.assertEqual(result["ignored_subdir"]["a.jpg"], "")
        self.assertEqual(result["ignored_subdir"]["b.jpg"], "")

        # Assert normal subdir file was processed once
        mock_export.assert_called_once()


    def test_prints_image_count_and_mean_duration_when_images_processed(self):
        """
        When images are processed, the function prints total image count and
        mean duration.
        """
        # create files by subdirectory with one image
        files_by_subdir = {"subdir1": ["img1.jpg"]}

        # Mock functions involved in processing
        with patch(
            "src.pipeline.extraction.os.path.exists", return_value=False
        ), \
            patch(
                "src.pipeline.extraction.func.preprocess_black_text",
                return_value="proc_img"
            ), \
            patch(
                "src.pipeline.extraction.func.resize_image",
                return_value="resized_img"
            ), \
            patch(
                "src.pipeline.extraction.func.extract_text_easyocr",
                return_value=("raw", 0.1)
            ), \
            patch("src.pipeline.extraction.func.export_text_to_json"), \
            patch("src.pipeline.extraction.tqdm", new=lambda x, **kw: x), \
            patch(
                "src.pipeline.extraction.config.IS_CORRECT_TEXT_FRENCH",
                False
            ), \
            patch(
                "src.pipeline.extraction.config.best_params",
                new={
                    "scale_percent": 50,
                    "batch_size": 1,
                    "decoder": "greedy",
                    "adjust_contrast": False,
                    "worker": 1
                }
            ), \
            patch("src.pipeline.extraction.config.USE_GPU_FOR_OCR", False), \
            patch("src.pipeline.extraction.time.time", return_value=2.0):

            # Capture printed output
            captured = StringIO()
            original_stdout = sys.stdout

            # Act
            try:
                sys.stdout = captured
                result = extract_text_from_images(
                    self.data_dir,
                    self.output_json_dir,
                    self.temp_dir,
                    files_by_subdir,
                    [],
                    "",
                    0.0
                )
            finally:
                sys.stdout = original_stdout

        # Assert printed output contains expected information
        output = captured.getvalue()
        self.assertIn("Nombre total d'images traitées : 1", output)
        self.assertIn("Durée moyenne / image", output)


if __name__ == "__main__":
    unittest.main()
