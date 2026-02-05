"""
Module de test unitaire pour le module src.pipeline.database.
"""
# imports for testing
import unittest
from unittest.mock import patch, Mock

# import the function to be tested
from src.pipeline.database import insert_into_database


class TestDatabaseModule(unittest.TestCase):
    """
    Unit tests for src.pipeline.database.
    """

    # patch the helper functions used in insert_into_database
    @patch("src.pipeline.database.func.insert_mesure_into_db")
    @patch("src.pipeline.database.func.insert_cheminee_into_db")
    @patch("src.pipeline.database.func.insert_client_into_db")
    @patch("src.pipeline.database.func.create_database_and_tables")
    @patch("src.pipeline.database.func.print_step")
    def test_insert_into_database_creates_db_and_inserts_data(
        self,
        mock_print_step,
        mock_create_db,
        mock_insert_client,
        mock_insert_cheminee,
        mock_insert_mesure
    ):
        """
        Ensure insert_into_database creates DB and calls all insertion
        helpers in correct order.
        """
        # Arrange test data
        bdd_dir = "/bdd"
        client_acronym = "ABC"
        client_name = "Client ABC"
        data_per_chimney = {"chimney1": [], "chimney2": []}
        date_mesure = "01/08/2025"

        # Arrange the mock of create_database_and_tables
        expected_bdd_path = "/bdd/database.db"
        mock_create_db.return_value = expected_bdd_path

        # Act
        result = insert_into_database(
            bdd_dir,
            client_acronym,
            client_name,
            data_per_chimney,
            date_mesure
        )

        # Assert the returned database path is correct
        self.assertEqual(result, expected_bdd_path)

        # Assert print_step was called with correct step number
        mock_print_step.assert_called_once_with(
            6, "Insertion des données dans la base de données"
        )

        # Assert create_database_and_tables was called with correct directory
        mock_create_db.assert_called_once_with(bdd_dir)

        # Assert all insertion helpers were called once with correct arguments
        mock_insert_client.assert_called_once_with(
            expected_bdd_path, client_acronym, client_name
        )
        mock_insert_cheminee.assert_called_once_with(
            expected_bdd_path, client_acronym, data_per_chimney
        )
        mock_insert_mesure.assert_called_once_with(
            expected_bdd_path,
            client_acronym,
            data_per_chimney,
            date_mesure
        )


    # patch the helper functions used in insert_into_database
    @patch("src.pipeline.database.func.insert_mesure_into_db")
    @patch("src.pipeline.database.func.insert_cheminee_into_db")
    @patch("src.pipeline.database.func.insert_client_into_db")
    @patch("src.pipeline.database.func.create_database_and_tables")
    @patch("src.pipeline.database.func.print_step")
    def test_insert_into_database_returns_database_path(
        self,
        mock_print_step,
        mock_create_db,
        mock_insert_client,
        mock_insert_cheminee,
        mock_insert_mesure
    ):
        """
        Ensure the returned value is the database path from
        create_database_and_tables.
        """
        # Arrange the mock of create_database_and_tables
        expected_path = "/backup/bdd/db_2025.db"
        mock_create_db.return_value = expected_path

        # Act
        result = insert_into_database(
            "/bdd", "XYZ", "Client XYZ", {}, "01/01/2025"
        )

        # Assert result matches expected path and is not None
        self.assertEqual(result, expected_path)
        self.assertIsNotNone(result)


    # patch the helper functions used in insert_into_database
    @patch("src.pipeline.database.func.insert_mesure_into_db")
    @patch("src.pipeline.database.func.insert_cheminee_into_db")
    @patch("src.pipeline.database.func.insert_client_into_db")
    @patch("src.pipeline.database.func.create_database_and_tables")
    @patch("src.pipeline.database.func.print_step")
    def test_insert_into_database_passes_correct_data_to_helpers(
        self,
        mock_print_step,
        mock_create_db,
        mock_insert_client,
        mock_insert_cheminee,
        mock_insert_mesure
    ):
        """
        Verify that all input parameters are correctly forwarded to the
        respective helper functions.
        """
        # Arrange test data
        bdd_dir = "/bdd"
        client_acronym = "DEF"
        client_name = "Company DEF"
        data_per_chimney = {"ch1": ["data1"], "ch2": ["data2"]}
        date_mesure = "15/06/2025"

        # Arrange the mock of create_database_and_tables
        bdd_path = "/bdd/db.db"
        mock_create_db.return_value = bdd_path

        # Act
        insert_into_database(
            bdd_dir, client_acronym, client_name, data_per_chimney, date_mesure
        )

        # Assert each helper function was called with the correct arguments
        client_call_args = mock_insert_client.call_args[0]
        self.assertEqual(client_call_args[0], bdd_path)
        self.assertEqual(client_call_args[1], client_acronym)
        self.assertEqual(client_call_args[2], client_name)

        chimney_call_args = mock_insert_cheminee.call_args[0]
        self.assertEqual(chimney_call_args[0], bdd_path)
        self.assertEqual(chimney_call_args[1], client_acronym)
        self.assertEqual(chimney_call_args[2], data_per_chimney)

        mesure_call_args = mock_insert_mesure.call_args[0]
        self.assertEqual(mesure_call_args[0], bdd_path)
        self.assertEqual(mesure_call_args[1], client_acronym)
        self.assertEqual(mesure_call_args[2], data_per_chimney)
        self.assertEqual(mesure_call_args[3], date_mesure)

        print_step_call_args = mock_print_step.call_args[0]
        self.assertEqual(print_step_call_args[0], 6)
        self.assertEqual(
            print_step_call_args[1],
            "Insertion des données dans la base de données"
        )


if __name__ == "__main__":
    unittest.main()
