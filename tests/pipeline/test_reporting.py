"""
Module de test unitaire pour le module src.pipeline.reporting.
"""
# imports standard
import os

# imports for testing
import unittest
from unittest.mock import patch

# import the function to be tested
from src.pipeline.reporting import generate_word_report


class TestReportingModule(unittest.TestCase):
    """
    Unit tests for src.pipeline.reporting.
    """

    # patch the helper functions used in generate_word_report
    @patch("src.pipeline.reporting.func.generate_word_report")
    @patch("src.pipeline.reporting.func.print_step")
    def test_generate_word_report_calls_helper_with_correct_parameters(
        self,
        mock_print_step,
        mock_generate_word_report
    ):
        """
        Ensure generate_word_report calls func.generate_word_report with
        correct arguments.
        """
        # Arrange test data
        data_per_chimney = {"chimney1": [], "chimney2": []}
        data_dir = "/data"
        output_dir = "/output"
        temp_dir = "/temp"
        client_name = "Client ABC"
        files_by_subdir = {"subdir1": ["img1.jpg"]}
        date_mesure = "01/08/2025"
        logo_dir = "/logos"
        expected_output_path = "/output/Annexes photo 3CEP-Client ABC.docx"

        # Configure mock
        mock_generate_word_report.return_value = expected_output_path

        # Act
        result = generate_word_report(
            data_per_chimney,
            data_dir,
            output_dir,
            temp_dir,
            client_name,
            files_by_subdir,
            date_mesure,
            logo_dir
        )

        # Assert return value
        self.assertEqual(result, expected_output_path)

        # Assert  helper function called with expected arguments
        mock_print_step.assert_called_once_with(
            5, "Génération du rapport Word"
        )
        mock_generate_word_report.assert_called_once()
        
        # Assert call arguments
        call_args = mock_generate_word_report.call_args
        self.assertEqual(call_args[0][0], data_per_chimney)
        self.assertEqual(call_args[0][1], data_dir)
        self.assertEqual(call_args[0][2], output_dir)
        self.assertEqual(call_args[0][3], temp_dir)
        self.assertEqual(call_args[0][4], client_name)
        self.assertEqual(call_args[0][5], files_by_subdir)
        self.assertEqual(call_args[0][6], date_mesure)
        
        # Assert good keyword arguments
        self.assertEqual(
            call_args[1]["logo_path"],
            os.path.join(logo_dir, "logo_rpur.png")
            )
        self.assertEqual(
            call_args[1]["output_file_name"],
            f"Annexes photo 3CEP-{client_name}.docx"
        )


    # patch the helper functions used in generate_word_report
    @patch("src.pipeline.reporting.func.generate_word_report")
    @patch("src.pipeline.reporting.func.print_step")
    def test_generate_word_report_returns_path_from_helper(
        self,
        mock_print_step,
        mock_generate_word_report
    ):
        """
        Ensure the function returns the path returned by func.generate_word_report.
        """
        # Arrange test data
        expected_path = "/backup/reports/report_2025.docx"
        mock_generate_word_report.return_value = expected_path

        # Act
        result = generate_word_report(
            {},
            "/data",
            "/output",
            "/temp",
            "Test Client",
            {},
            "01/01/2025",
            "/logos"
        )

        # Assert
        self.assertEqual(result, expected_path)


    # patch the helper functions used in generate_word_report
    @patch("src.pipeline.reporting.func.generate_word_report")
    @patch("src.pipeline.reporting.func.print_step")
    def test_generate_word_report_constructs_correct_filename(
        self,
        mock_print_step,
        mock_generate_word_report
    ):
        """
        Verify that the output filename follows the expected format.
        """
        # Arrange test data
        client_name = "ACME Corp"
        mock_generate_word_report.return_value = "/output/report.docx"

        # Act
        generate_word_report(
            {},
            "/data",
            "/output",
            "/temp",
            client_name,
            {},
            "01/01/2025",
            "/logos"
        )

        # Assert the output_file_name parameter matches expected format
        call_kwargs = mock_generate_word_report.call_args[1]
        expected_filename = f"Annexes photo 3CEP-{client_name}.docx"
        self.assertEqual(call_kwargs["output_file_name"], expected_filename)


    # patch the helper functions used in generate_word_report
    @patch("src.pipeline.reporting.func.generate_word_report")
    @patch("src.pipeline.reporting.func.print_step")
    def test_generate_word_report_constructs_correct_logo_path(
        self,
        mock_print_step,
        mock_generate_word_report
    ):
        """
        Verify that the logo path is correctly constructed from logo_dir.
        """
        # Arrange test data
        logo_dir = "/images/assets"
        mock_generate_word_report.return_value = "/output/report.docx"

        # Act
        generate_word_report(
            {},
            "/data",
            "/output",
            "/temp",
            "Client",
            {},
            "01/01/2025",
            logo_dir
        )

        # Assert the logo_path parameter is correctly constructed
        call_kwargs = mock_generate_word_report.call_args[1]
        expected_logo_path = os.path.join(logo_dir, "logo_rpur.png")
        self.assertEqual(call_kwargs["logo_path"], expected_logo_path)


if __name__ == "__main__":
    unittest.main()
