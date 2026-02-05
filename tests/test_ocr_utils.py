"""
Module de tests unitaires pour les fonctions OCR et utilitaires associées.
"""
# Import standard libraries
import os
import json
import tempfile
import shutil
import sqlite3
import numpy as np
from io import StringIO

# Imports for testing
import unittest
from unittest.mock import patch, MagicMock, mock_open

# import modules to test
import src.config as config
from src.ocr_utils import (
    Logger,
    get_git_version,
    format_git_version,
    check_and_create_directories,
    preprocess_black_text,
    resize_image,
    compress_image,
    extract_text_easyocr,
    correct_text_french,
    export_text_to_json,
    clean_text,
    extract_key_info,
    generate_filename,
    copy_files_with_mapping,
    import_json_to_text,
    sort_key,
    group_by_chimney_name,
    save_to_json,
    get_client_name_counts,
    add_page_number_field,
    add_picture_to_paragraph,
    generate_word_report,
    create_database_and_tables,
    insert_client_into_db,
    insert_cheminee_into_db,
    insert_mesure_into_db,
    print_step,
    calculate_duration,
    backup_database,
    backup_word_report,
    backup_json,
    measure_time
)


class TestLogger(unittest.TestCase):
    """
    Tests pour la classe Logger.
    """
    # patch the open function to avoid actual file I/O
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_initialization(self, mock_file):
        """
        Test to initalization of Logger.
        """
        logger = Logger("/path/to/log.txt")
        self.assertIsNotNone(logger.terminal)
        mock_file.assert_called_once_with(
            "/path/to/log.txt", "a", encoding="utf-8"
        )

    # patch the open function to avoid actual file I/O
    @patch('builtins.open', new_callable=mock_open)
    def test_logger_write(self, mock_file):
        """
        Test write of Logger.
        """
        logger = Logger("/path/to/log.txt")
        logger.write("Test message")
        mock_file().write.assert_called_with("Test message")


class TestGitVersionFunctions(unittest.TestCase):
    """
    Tests pour les fonctions de version Git.
    """
    # patch subprocess.check_output to simulate git command output
    @patch('subprocess.check_output')
    def test_get_git_version_success(self, mock_subprocess):
        """
        Test of Git version retrieval success.
        """
        mock_subprocess.return_value = b"v1.0.0-5-g1234abc\n"
        version = get_git_version()
        self.assertEqual(version, "v1.0.0-5-g1234abc")
        mock_subprocess.assert_called_once()


    # patch subprocess.check_output to simulate git command failure
    @patch('subprocess.check_output')
    def test_get_git_version_failure(self, mock_subprocess):
        """
        Test of Git version retrieval failure.
        """
        mock_subprocess.side_effect = Exception("Git error")
        version = get_git_version()
        self.assertEqual(version, "version inconnue")


    def test_format_git_version_tag_only(self):
        """
        Test to format Git version with tag only.
        """
        result = format_git_version("v1.0.0")
        self.assertEqual(result, "Version : v1.0.0")


    def test_format_git_version_full(self):
        """
        Test to format full Git version.
        """
        result = format_git_version("v1.0.0-5-g1234abc")
        expected = "Version : v1.0.0 (5 commits après le tag, commit g1234abc)"
        self.assertEqual(result, expected)


class TestDirectoryManagement(unittest.TestCase):
    """Tests for directory management functions."""

    # patch os.path.exists and os.makedirs for check_and_create_directories
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_check_and_create_directories_creates_missing(
        self, mock_makedirs, mock_exists
    ):
        """
        Test of check_and_create_directories creates missing directories.
        """      
        mock_exists.return_value = False
        check_and_create_directories("/dir1", "/dir2", "/dir3")
        self.assertEqual(mock_makedirs.call_count, 3)
        mock_makedirs.assert_any_call("/dir1", exist_ok=True)
        mock_makedirs.assert_any_call("/dir2", exist_ok=True)
        mock_makedirs.assert_any_call("/dir3", exist_ok=True)


    # patch os.path.exists and os.makedirs for check_and_create_directories
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_check_and_create_directories_skips_existing(
        self, mock_makedirs, mock_exists
    ):
        """
        Test that check_and_create_directories skips existing directories.
        """      
        mock_exists.return_value = True
        check_and_create_directories("/existing_dir")
        mock_makedirs.assert_not_called()


class TestImageProcessing(unittest.TestCase):
    """Tests for image processing functions."""
    
    # patch multiple cv2 and PIL functions for preprocess_black_text
    @patch('cv2.imwrite')
    @patch('cv2.bitwise_and')
    @patch('cv2.inRange')
    @patch('cv2.cvtColor')
    @patch('PIL.Image.open')
    @patch('numpy.array')
    def test_preprocess_black_text(
        self,
        mock_np_array,
        mock_image_open,
        mock_cvtcolor,
        mock_inrange,
        mock_bitwise_and,
        mock_imwrite
    ):
        """
        Test of preprocess_black_text with valid image processing flow.
        """     
        # Mock image_open to return a mock image object
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        # Mock numpy.array for returning a dummy image
        H, W = 100, 100
        dummy_image = np.ones((H, W, 3), dtype=np.uint8) * 128
        mock_np_array.return_value = dummy_image

        # Mock cv2.cvtColor for return a dummy HSV image
        hsv_image = np.ones((H, W, 3), dtype=np.uint8) * 128
        mock_cvtcolor.return_value = hsv_image
        
        # Mock cv2.inRange for return a dummy mask
        mask = np.zeros((H, W), dtype=np.uint8)
        mock_inrange.return_value = mask
        
        # Mock cv2.bitwise_and for return a dummy result image
        result_image = np.ones((H, W, 3), dtype=np.uint8) * 255
        mock_bitwise_and.return_value = result_image
        
        # act
        result = preprocess_black_text("/input.jpg", "/output.jpg")
        
        # Assert result is not None and has expected shape
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (H, W, 3))
        
        # Assert that the expected functions were called with correct parameters
        mock_image_open.assert_called_once_with("/input.jpg")
        mock_cvtcolor.assert_called_once()
        mock_inrange.assert_called_once()
        mock_bitwise_and.assert_called_once()
        mock_imwrite.assert_called_once_with("/output.jpg", result)


    # patch PIL.Image.open to simulate an error for preprocess_black_text
    @patch('PIL.Image.open')
    def test_preprocess_black_text_error_handling(self, mock_image_open):
        """
        Test of preprocess_black_text error handling when image loading fails.
        """
        mock_image_open.side_effect = Exception("Image load error")
        result = preprocess_black_text("/invalid.jpg", "/output.jpg")
        self.assertIsNone(result)


    # patch PIL and cv2 functions used in resize_image
    @patch('PIL.Image.open')
    @patch('cv2.resize')
    @patch('cv2.imwrite')
    def test_resize_image(self, mock_imwrite, mock_resize, mock_image_open):
        """
        Test of resize_image function.
        """
        # Mock image open to return a mock image object
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        # Mock numpy array for returning a dummy image and act
        mock_array = np.ones((1000, 1000, 3), dtype=np.uint8)
        with patch('numpy.array', return_value=mock_array):
            result = resize_image(
                "/input.jpg", "/output.jpg", scale_percent=50
            )
        
        # Assert expected functions were called
        mock_resize.assert_called_once()
        mock_imwrite.assert_called_once()


    # patch PIL and cv2 functions used in compress_image
    @patch('PIL.Image.open')
    @patch('cv2.resize')
    @patch('cv2.cvtColor')
    @patch('cv2.imwrite')
    def test_compress_image(
        self, mock_imwrite, mock_cvtcolor, mock_resize, mock_image_open
    ):
        """
        Test of compress_image function with valid image processing flow.
        """
        # Mock image open to return a mock image object
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value.__enter__.return_value = mock_img

        # Mock numpy array for returning a large dummy image and act
        mock_array = np.ones((1200, 1600, 3), dtype=np.uint8)
        with patch('numpy.array', return_value=mock_array):
            result = compress_image(
                "/input.jpg",
                "/temp",
                max_width=800,
                max_height=800,
                quality=50
            )

        # Assert result is not None and other functions were called 
        self.assertIsNotNone(result)
        mock_resize.assert_called_once()
        mock_imwrite.assert_called_once()


class TestOCRFunctions(unittest.TestCase):
    """
    Tests for OCR-related functions, including text extraction and correction.
    """

    # patch easyocr.Reader to simulate extraction for extract_text_easyocr
    @patch('easyocr.Reader')
    def test_extract_text_easyocr(self, mock_reader):
        """
        Test of extract_text_easyocr with mocked OCR reader.
        """
        # Mock instance with a predefined readtext output
        mock_instance = MagicMock()
        mock_instance.readtext.return_value = ["ligne 1", "ligne 2", "a1 b2"]
        mock_reader.return_value = mock_instance
        
        # Act
        text, duration = extract_text_easyocr(
            "/image.jpg",
            batch_size=1,
            decoder="wordbeamsearch",
            adjust_contrast=0.5,
            worker=0,
            gpu_state=False
        )
        
        # Assert expected results
        self.assertIn("ligne 1", text)
        self.assertIn("ligne 2", text)
        self.assertIn("a1 b2", text)
        self.assertIsInstance(duration, float)


    # patch LanguageTool to simulate correction for correct_text_french
    @patch('language_tool_python.LanguageTool')
    def test_correct_text_french(self, mock_language_tool):
        """
        Test of correct_text_french with mocked LanguageTool.
        """
        # Mock instance with a predefined correct output
        mock_tool = MagicMock()
        mock_tool.correct.return_value = "texte corrigé"
        mock_language_tool.return_value = mock_tool

        # Act
        corrected, duration = correct_text_french("texte incorect")

        # Assert expected results
        self.assertEqual(corrected, "texte corrigé")
        self.assertIsInstance(duration, float)


class TestTextCleaning(unittest.TestCase):
    """Tests for text cleaning functions."""

    def test_clean_text_removes_newlines(self):
        """
        Test of newline removal in text cleaning.
        """
        text = "ligne 1\nligne 2\rligne 3"
        result = clean_text(text)
        
        self.assertNotIn("\n", result)
        self.assertNotIn("\r", result)
        self.assertIn("ligne 1 ligne 2 ligne 3", result)


    def test_clean_text_removes_special_chars(self):
        """
        Test of special character removal in text cleaning.
        """
        text = "texte@#$%avec&*()caractères"
        result = clean_text(text)
        self.assertNotIn("@", result)
        self.assertNotIn("#", result)
        self.assertNotIn("&", result)


    def test_clean_text_converts_to_lowercase(self):
        """
        Test of conversion to lowercase.
        """
        text = "TEXTE EN MAJUSCULES"
        result = clean_text(text)
        self.assertEqual(result, "texte en majuscules")


    def test_clean_text_handles_tuple(self):
        """
        Test of cleaning with tuple input.
        """
        text = ("partie1", "partie2", "partie3")
        result = clean_text(text)
        self.assertIsInstance(result, str)
        self.assertIn("partie1", result)


class TestKeyInfoExtraction(unittest.TestCase):
    """
    Tests for key information extraction.
    """

    def test_extract_key_info_with_matches(self):
        """
        Test of extraction with matches.
        """
        text = "Client Test a1 b2 c3 remarque importante"
        result = extract_key_info(text)
        self.assertEqual(result["client_name"], "Client Test")
        self.assertIn("a1", result["chimney_name"])
        self.assertIn("b2", result["chimney_name"])
        self.assertIn("c3", result["chimney_name"])
        self.assertEqual(result["remarks"], "remarque importante")


    def test_extract_key_info_no_matches(self):
        """
        Test of extraction without matches.
        """
        text = "Texte sans correspondances"
        result = extract_key_info(text)
        self.assertEqual(result["client_name"], "")
        self.assertEqual(result["chimney_name"], "")
        self.assertEqual(result["remarks"], "")


class TestFilenameGeneration(unittest.TestCase):
    """
    Tests for filename generation.
    """

    def test_generate_filename_new_file(self):
        """
        Test of new filename generation.
        """
        text = "Client a1"
        pattern = config.pattern
        filenames = []
        new_filenames, key_info = generate_filename(
            text, pattern, "CLIENT", filenames
        )
        self.assertEqual(new_filenames[0], "CLIENT_a1.jpg")


    def test_generate_filename_duplicate_handling(self):
        """
        Test of duplicate handling.
        """
        text = "Client a1"
        pattern = config.pattern
        filenames = ["CLIENT_a1.jpg"]
        new_filenames, key_info = generate_filename(
            text, pattern, "CLIENT", filenames
        )
        self.assertEqual(new_filenames[0], "CLIENT_a1 (1).jpg")


    def test_generate_filename_no_chimney(self):
        """
        Test of filename generation without chimney name.
        """
        text = "Texte sans cheminée"
        pattern = config.pattern
        filenames = []
        new_filenames, key_info = generate_filename(
            text, pattern, "CLIENT", filenames
        )
        self.assertEqual(new_filenames[0], "Same_as_original")


class TestJSONOperations(unittest.TestCase):
    """
    Tests for JSON operations.
    """

    # patch built-in open and json.dump for export_text_to_json
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_export_text_to_json(self, mock_json_dump, mock_file):
        """
        Test of exporting to JSON.
        """
        data = {"key": "value"}
        export_text_to_json(data, "/output/dir", "test.json")
        mock_file.assert_called_once()
        mock_json_dump.assert_called_once()


    # patch built-in open and json.load for import_json_to_text
    @patch(
            'builtins.open',
            new_callable=mock_open,
            read_data='{"key": "value"}'
        )
    @patch('json.load')
    def test_import_json_to_text(self, mock_json_load, mock_file):
        """
        Test of importing from JSON.
        """
        mock_json_load.return_value = {"key": "value"}
        result = import_json_to_text("/input/dir", "test.json")
        self.assertEqual(result, {"key": "value"})
        mock_file.assert_called_once()


    # patch built-in open to simulate FileNotFoundError for import_json_to_text
    @patch('builtins.open', new_callable=mock_open)
    def test_import_json_to_text_file_not_found(self, mock_file):
        """
        Test of importing with missing file.
        """
        mock_file.side_effect = FileNotFoundError()
        result = import_json_to_text("/input/dir", "missing.json")
        self.assertIsNone(result)


    # patch built-in open and json.dump for save_to_json
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_to_json(self, mock_json_dump, mock_file):
        """
        Test of saving to JSON.
        """
        data = {"test": "data"}
        result = save_to_json(data, "/output/dir", "test.json")
        self.assertIsNotNone(result)
        mock_json_dump.assert_called_once()


class TestDataGrouping(unittest.TestCase):
    """
    Tests for data grouping.
    """

    def test_sort_key_with_match(self):
        """
        Test of sorting key with match.
        """
        pattern = config.pattern
        result = sort_key("a1", pattern)
        self.assertEqual(result, [("a", 1)])


    def test_sort_key_no_match(self):
        """
        Test of sorting key without match.
        """
        pattern = config.pattern
        result = sort_key("test", pattern)
        self.assertEqual(result, [("test", 0)])


    def test_group_by_chimney_name(self):
        """
        Test of grouping by chimney name.
        """
        data = {
            "subdir1": {
                "file1.jpg": {
                    "client_name": "Client A",
                    "chimney_name": ["a1"],
                    "remarks": "Remarque 1"
                }
            }
        }
        result = group_by_chimney_name(data)
        self.assertIn("a1", result)
        self.assertEqual(len(result["a1"]), 1)
        self.assertEqual(result["a1"][0]["file"], "file1.jpg")


    def test_group_by_chimney_name_empty_chimney(self):
        """
        Test of grouping with empty chimney.
        """
        data = {
            "subdir1": {
                "file1.jpg": {
                    "client_name": "Client A",
                    "chimney_name": [],
                    "remarks": ""
                }
            }
        }
        result = group_by_chimney_name(data)
        self.assertTrue(any("No_chimney" in key for key in result.keys()))


class TestClientNameExtraction(unittest.TestCase):
    """
    Tests for client name extraction.
    """

    def test_get_client_name_counts(self):
        """
        Test of counting client names.
        """
        data = {
            "subdir1": {
                "file1.jpg": {"client_name": "Client A", "remarks": ""},
                "file2.jpg": {"client_name": "Client A", "remarks": ""}
            },
            "subdir2": {
                "file3.jpg": {"client_name": "Client B", "remarks": ""}
            }
        }
        result = get_client_name_counts(data)
        self.assertEqual(result["Client A"], 2)
        self.assertEqual(result["Client B"], 1)


    def test_get_client_name_counts_ignores_empty(self):
        """
        Test that empty client names are ignored.
        """
        data = {
            "subdir1": {
                "file1.jpg": {"client_name": "", "remarks": ""},
                "file2.jpg": {"client_name": "Client A", "remarks": ""}
            }
        }
        result = get_client_name_counts(data)
        self.assertNotIn("", result)
        self.assertEqual(result["Client A"], 1)


class TestDatabaseOperations(unittest.TestCase):
    """
    Tests for database operations.
    """

    def setUp(self):
        """
        Create a temporary database for testing.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")


    def tearDown(self):
        """
        Clean up after tests.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    def test_create_database_and_tables(self):
        """
        Test of creating database and tables.
        """
        # Create database and  assert it exists
        result = create_database_and_tables(self.temp_dir, "test.db")
        self.assertTrue(os.path.exists(result))

        # Check that the expected tables exist
        conn = sqlite3.connect(result)
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cur.fetchall()]
        conn.close()

        # Assert that the expected tables are present
        self.assertIn("clients", tables)
        self.assertIn("cheminees", tables)
        self.assertIn("mesures", tables)


    def test_insert_client_into_db(self):
        """
        Test of inserting a client.
        """
        # Create database and insert a client
        db_path = create_database_and_tables(self.temp_dir, "test.db")
        insert_client_into_db(db_path, "ABC", "Client Test")
        
        # Check that the client was inserted correctly
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM clients WHERE client_id = ?", ("ABC",))
        result = cur.fetchone()
        conn.close()
        
        # Assert that the result is not None and has expected values
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "ABC")
        self.assertEqual(result[1], "Client Test")


    def test_insert_cheminee_into_db(self):
        """
        Test of inserting a chimney.
        """
        # Create database, insert a client, and then insert a chimney
        db_path = create_database_and_tables(self.temp_dir, "test.db")
        insert_client_into_db(db_path, "ABC", "Client Test")
        data_per_chimney = {
            "a1": [
                {"subdir": "subdir1", "remarks": "test remark"}
            ]
        }
        insert_cheminee_into_db(db_path, "ABC", data_per_chimney)

        # Check that the chimney was inserted correctly
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM cheminees WHERE cheminee_id = ?", ("a1",))
        result = cur.fetchone()
        conn.close()

        # Assert that the result is not None and has expected values
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "a1")


    def test_insert_mesure_into_db(self):
        """
        Test of inserting a measure.
        """
        # Create database, insert a client and a chimney, then insert a measure
        db_path = create_database_and_tables(self.temp_dir, "test.db")
        insert_client_into_db(db_path, "ABC", "Client Test")
        data_per_chimney = {
            "a1": [{"subdir": "subdir1", "remarks": ""}]
        }
        insert_cheminee_into_db(db_path, "ABC", data_per_chimney)
        insert_mesure_into_db(db_path, "ABC", data_per_chimney, "15/01/2025")

        # Check that the measure was inserted correctly
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT * FROM mesures WHERE cheminee_id = ?", ("a1",))
        result = cur.fetchone()
        conn.close()
        
        # Assert that the result is not None and has expected values
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "a1")


class TestWordReportGeneration(unittest.TestCase):
    """ 
    Tests for Word report generation.
    """

    # patch the helper functions used in generate_word_report
    @patch("os.makedirs")
    @patch("os.path.exists")
    @patch("src.ocr_utils.add_picture_to_paragraph")
    @patch("src.ocr_utils.compress_image")
    @patch("src.ocr_utils.Document")
    def test_generate_word_report_creates_document(
        self,
        mock_document,
        mock_compress_image,
        mock_add_pic,
        mock_exists,
        mock_makedirs
    ):
        """ 
        Test of generate_word_report creates a Word document.
        """
        # mock os.path.exists & os.makedirs to simulate directory/file exist
        mock_exists.side_effect = lambda path: True
        mock_makedirs.return_value = None

        # Mock Document and its methods
        mock_doc = MagicMock()
        mock_document.return_value = mock_doc
        # Header mock
        mock_header_paragraph = MagicMock()
        mock_doc.sections = [
            MagicMock(
                header=MagicMock(paragraphs=[mock_header_paragraph]),
                footer=MagicMock(paragraphs=[MagicMock()])
            )
        ]
        # Mock add_run and add_picture for header
        mock_run = MagicMock()
        mock_header_paragraph.add_run.return_value = mock_run
        mock_run.add_picture.return_value = None
        # Main title and chimney paragraphs
        mock_doc.add_paragraph.return_value = MagicMock(
            add_run=MagicMock(return_value=MagicMock())
        )
        # Table mock
        mock_doc.add_table.return_value = MagicMock(
            cell=MagicMock(
                return_value=MagicMock(
                    add_paragraph=MagicMock(return_value=MagicMock())
                )
            ),
            style=None,
            autofit=None,
            columns=[MagicMock(width=None), MagicMock(width=None)]
        )

        # Mock compress_image to return a dummy path
        mock_compress_image.side_effect = (
            lambda path, temp_dir: path + "_compressed"
        )

        # Mock add_picture_to_paragraph to do nothing
        mock_add_pic.return_value = None

        # Test data
        data_per_chimney = {
            "a1": [{
                "subdir": "subdir1",
                "file": "file1.jpg",
                "client_name": "Test"
            }]
        }
        files_by_subdir = {"subdir1": ["file1.jpg"]}

        # Act
        result = generate_word_report(
            data_per_chimney,
            "/input",
            "/output",
            "/temp",
            "Client Test",
            files_by_subdir,
            "15/01/2025",
            "/logo.png"
        )

        # Assert that a document was created and saved
        self.assertIsNotNone(result)

        # Assert that the expected functions were called with correct parameters
        mock_document.assert_called_once()
        mock_doc.save.assert_called_once_with(result)
        mock_run.add_picture.assert_called_once_with(
            "/logo.png", width=unittest.mock.ANY
        )
        mock_compress_image.assert_called()
        mock_add_pic.assert_called()


class TestBackupOperations(unittest.TestCase):
    """
    Tests for backup operations.
    """

    def setUp(self):
        """
        Create temporary directories for testing.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.backup_dir = os.path.join(self.temp_dir, "backup")


    def tearDown(self):
        """
        Clean up after tests.
        """
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


    # patch the helper functions used in backup_database
    @patch('os.listdir')
    @patch('os.path.getctime')
    @patch('shutil.copy')
    @patch('os.makedirs')
    def test_backup_database(
        self, mock_makedirs, mock_copy, mock_getctime, mock_listdir
    ):
        """
        Test of database backup with no existing backups.
        """
        mock_listdir.return_value = []
        mock_getctime.return_value = 1234567890.0
        backup_database("/db/test.db", self.backup_dir)
        mock_makedirs.assert_called_once()
        mock_copy.assert_called_once()


    # patch the helper functions used in backup_word_report
    @patch('shutil.copy')
    @patch('os.makedirs')
    def test_backup_word_report(self, mock_makedirs, mock_copy):
        """
        Test of backing up a Word report.
        """
        backup_word_report("/word/report.docx", self.backup_dir, "ABC")
        mock_makedirs.assert_called_once()
        mock_copy.assert_called_once()


    # patch the helper functions used in backup_json
    @patch('os.listdir')
    @patch('shutil.copy')
    @patch('os.makedirs')
    def test_backup_json(self, mock_makedirs, mock_copy, mock_listdir):
        """
        Test of backing up JSON files.
        """
        mock_listdir.return_value = ["file1.json", "file2.json", "file3.txt"]
        backup_json("/json/path", self.backup_dir, "ABC")
        self.assertEqual(mock_makedirs.call_count, 2)
        self.assertEqual(mock_copy.call_count, 2)


class TestUtilityFunctions(unittest.TestCase):
    """
    Tests for utility functions.
    """

    # patch the helper functions used in print_step
    @patch('sys.stdout', new_callable=StringIO)
    def test_print_step(self, mock_stdout):
        """
        Test of printing steps.
        """
        print_step(1, "Test message")
        output = mock_stdout.getvalue()
        self.assertIn("STEP 1", output)
        self.assertIn("Test message", output)
        self.assertIn("*" * 75, output)


    # patch the helper function used in calculate_duration
    @patch('time.time')
    def test_calculate_duration(self, mock_time):
        """
        Test of calculating duration.
        """        
        mock_time.return_value = 200.0
        start_time = 0.0
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            calculate_duration(start_time)
            output = mock_stdout.getvalue()
        self.assertIn("3 min", output)
        self.assertIn("20 sec", output)


    def test_measure_time_decorator(self):
        """
        Test of the measure_time decorator.
        """  
        @measure_time
        def test_function():
            return "result"
        result, duration = test_function()
        self.assertEqual(result, "result")
        self.assertIsInstance(duration, float)
        self.assertGreaterEqual(duration, 0)


class TestFileCopyMapping(unittest.TestCase):
    """
    Tests for copying files with mapping.
    """

    # patch the helper functions used in copy_files_with_mapping
    @patch('os.makedirs')
    @patch('shutil.copy')
    @patch('src.ocr_utils.export_text_to_json')
    def test_copy_files_with_mapping(
        self, mock_export, mock_copy, mock_makedirs
    ):
        """
        Test of copying files with mapping.
        """
        # create test data and config
        text_extracted = {
            "subdir1": {
                "file1.jpg": "Client Test a1 remarques"
            }
        }
        pattern = config.pattern

        # act
        key_info_file, mapping_file = copy_files_with_mapping(
            text_extracted,
            pattern,
            "/input",
            "/output",
            "/json",
            "CLIENT"
        )

        # Assert that the expected files are created and functions are called
        self.assertEqual(key_info_file, "key_info.json")
        self.assertEqual(mapping_file, "file_mapping.json")
        self.assertEqual(mock_export.call_count, 2)
        mock_makedirs.assert_called()
        mock_copy.assert_called()


class TestWordDocumentHelpers(unittest.TestCase):
    """
    Tests for the helper functions for creating Word documents.
    """

    # patch the helper function used in add_page_number_field
    @patch('docx.oxml.OxmlElement')
    def test_add_page_number_field(self, mock_element):
        """
        Test of adding a page number field.
        """
        mock_paragraph = MagicMock()
        mock_run = MagicMock()
        mock_paragraph.add_run.return_value = mock_run
        add_page_number_field(mock_paragraph)
        mock_paragraph.add_run.assert_called()


    # patch the helper functions used in add_picture_to_paragraph
    @patch('docx.shared.Inches')
    def test_add_picture_to_paragraph_success(self, mock_inches):
        """
        Test of adding a picture to a paragraph (success).
        """
        mock_paragraph = MagicMock()
        mock_run = MagicMock()
        mock_paragraph.add_run.return_value = mock_run
        add_picture_to_paragraph(mock_paragraph, "/valid/image.jpg")
        mock_run.add_picture.assert_called_once()


    def test_add_picture_to_paragraph_none_path(self):
        """
        Test of adding a picture with a None path.
        """
        mock_paragraph = MagicMock()
        add_picture_to_paragraph(mock_paragraph, None)
        mock_paragraph.add_run.assert_called()


class TestEdgeCases(unittest.TestCase):
    """
    Tests for edge cases.
    """

    def test_clean_text_empty_string(self):
        """
        Test of cleaning with empty string.
        """
        result = clean_text("")
        self.assertEqual(result, "")


    def test_clean_text_only_special_chars(self):
        """
        Test of cleaning with only special characters.
        """
        result = clean_text("@#$%^&*()")
        self.assertEqual(result, "")


    def test_generate_filename_empty_text(self):
        """
        Test of generating filename with empty text.
        """
        result, key_info = generate_filename("", config.pattern, "CLIENT", [])
        self.assertEqual(result[0], "Same_as_original")


    def test_group_by_chimney_name_multiple_chimneys_same_file(self):
        """
        Test of grouping with multiple chimneys for a file.
        """
        data = {
            "subdir1": {
                "file1.jpg": {
                    "client_name": "Client A",
                    "chimney_name": ["a1", "a2", "a3"],
                    "remarks": ""
                }
            }
        }
        result = group_by_chimney_name(data)
        self.assertIn("a1", result)
        self.assertIn("a2", result)
        self.assertIn("a3", result)
        self.assertEqual(result["a1"][0]["file"], "file1.jpg")
        self.assertEqual(result["a2"][0]["file"], "file1.jpg")
        self.assertEqual(result["a3"][0]["file"], "file1.jpg")


class TestErrorHandling(unittest.TestCase):
    """
    Tests for error handling.
    """

    # patch the helper functions used in export_text_to_json
    @patch('builtins.open')
    def test_export_text_to_json_permission_error(self, mock_open):
        """
        Test of permission error during JSON export.
        """
        mock_open.side_effect = PermissionError()
        with patch('sys.stdout', new_callable=StringIO):
            export_text_to_json({}, "/readonly/dir", "test.json")
        self.assertTrue(True)


    # patch the helper functions used in import_json_to_text
    @patch('builtins.open')
    def test_import_json_invalid_json(self, mock_open):
        """
        Test of importing invalid JSON.
        """
        mock_open.return_value.__enter__.return_value.read.return_value = (
            "invalid json"
        )
        with patch(
            'json.load', side_effect=json.JSONDecodeError("msg", "doc", 0)
        ):
            with patch('sys.stdout', new_callable=StringIO):
                result = import_json_to_text("/dir", "invalid.json")
        self.assertIsNone(result)


    # patch the helper function used in insert_client_into_db
    @patch('sqlite3.connect')
    def test_database_connection_error(self, mock_connect):
        """
        Test of database connection error.
        """
        mock_connect.side_effect = sqlite3.Error("Connection failed")
        with patch('sys.stdout', new_callable=StringIO):
            insert_client_into_db("/invalid/db.db", "ABC", "Client")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
