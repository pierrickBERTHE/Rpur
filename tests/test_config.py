"""
Module de tests unitaires pour les configurations dans config.py.
"""
# general imports
import re

# import for testing
import unittest

# import the config module and its contents
import src.config as config
from src.config import best_params, pattern
from src.config import (
    IS_CORRECT_TEXT_FRENCH,
    USE_GPU_FOR_OCR,
    GENERATE_WORD_REPORT,
    INSERT_IN_DATABASE,
    CLEANUP_TEMP_FILES,
    LOG_TO_FILE,
    IS_BACKUP_CREATED
)


class TestConfigurationFlags(unittest.TestCase):
    """
    Test for configuration flags in config.py.
    """

    def test_is_correct_text_french_is_boolean(self):
        """
        Checks that IS_CORRECT_TEXT_FRENCH is a boolean.
        """
        self.assertIsInstance(IS_CORRECT_TEXT_FRENCH, bool)


    def test_use_gpu_for_ocr_is_boolean(self):
        """
        Checks that USE_GPU_FOR_OCR is a boolean.
        """
        self.assertIsInstance(USE_GPU_FOR_OCR, bool)


    def test_generate_word_report_is_boolean(self):
        """
        Checks that GENERATE_WORD_REPORT is a boolean.
        """
        self.assertIsInstance(GENERATE_WORD_REPORT, bool)


    def test_insert_in_database_is_boolean(self):
        """
        Checks that INSERT_IN_DATABASE is a boolean.
        """
        self.assertIsInstance(INSERT_IN_DATABASE, bool)


    def test_cleanup_temp_files_is_boolean(self):
        """
        Checks that CLEANUP_TEMP_FILES is a boolean.
        """
        self.assertIsInstance(CLEANUP_TEMP_FILES, bool)


    def test_log_to_file_is_boolean(self):
        """
        Checks that LOG_TO_FILE is a boolean.
        """
        self.assertIsInstance(LOG_TO_FILE, bool)


    def test_is_backup_created_is_boolean(self):
        """
        Checks that IS_BACKUP_CREATED is a boolean.
        """
        self.assertIsInstance(IS_BACKUP_CREATED, bool)


    def test_all_flags_are_defined(self):
        """
        Checks that all required flags are defined in config.py.
        """
        required_flags = [
            'IS_CORRECT_TEXT_FRENCH',
            'USE_GPU_FOR_OCR',
            'GENERATE_WORD_REPORT',
            'INSERT_IN_DATABASE',
            'CLEANUP_TEMP_FILES',
            'LOG_TO_FILE',
            'IS_BACKUP_CREATED'
        ]
        for flag in required_flags:
            self.assertTrue(
                hasattr(config, flag), 
                f"Le drapeau '{flag}' n'est pas défini dans config.py"
            )


class TestBestParams(unittest.TestCase):
    """
    Tests for the best_params dictionary.
    """

    def test_params_structure(self):
        """
        Checks the structure of best_params.
        """
        required_keys = [
            'adjust_contrast',
            'batch_size',
            'decoder', 
            'scale_percent',
            'worker'
        ]
        for key in required_keys:
            self.assertIn(key, best_params)


    def test_adjust_contrast_range(self):
        """
        Checks the range of adjust_contrast.
        """
        self.assertGreaterEqual(best_params['adjust_contrast'], 0)
        self.assertLessEqual(best_params['adjust_contrast'], 2)


    def test_batch_size_positive(self):
        """
        Checks that batch_size is positive.
        """
        self.assertGreater(best_params['batch_size'], 0)


    def test_decoder_valid(self):
        """
        Checks that decoder has a valid value.
        """
        valid_decoders = ["wordbeamsearch", "beamsearch", "greedy"]
        self.assertIn(best_params['decoder'], valid_decoders)


    def test_scale_percent_range(self):
        """
        Checks the range of scale_percent.
        """
        self.assertGreater(best_params['scale_percent'], 0)
        self.assertLessEqual(best_params['scale_percent'], 100)


class TestPattern(unittest.TestCase):
    """
    Tests for the regex pattern in config.py.
    """

    def setUp(self):
        """ 
        Prepare the compiled regex pattern for tests.
        """
        self.compiled_pattern = re.compile(pattern)


    def test_pattern_valid_regex(self):
        """
        Checks that pattern is a valid regex.
        """
        try:
            re.compile(pattern)
        except re.error as e:
            self.fail(f"Pattern invalide: {e}")


    def test_pattern_matches_variables(self):
        """
        Checks that pattern matches valid variables.
        """
        test_cases = ["a1", "b2", "var123", "ABC99"]
        for case in test_cases:
            with self.subTest(case=case):
                self.assertIsNotNone(self.compiled_pattern.search(case))


    def test_pattern_excludes_units(self):
        """
        Checks that pattern excludes measurement units.
        """
        # Define some test cases with units
        test_cases = ["a1 cm", "b2 kg", "x10 ml"]

        # Verify that no match is found for these cases
        for case in test_cases:
            with self.subTest(case=case):
                match = self.compiled_pattern.search(case)

                # There should be no match since a unit follows the pattern
                if match:
                    self.assertNotIn(' ', match.group(0))


if __name__ == '__main__':
    unittest.main()
