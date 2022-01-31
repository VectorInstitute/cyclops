"""Querier class for performing SQL queries."""

import time
import logging
import argparse

import pandas as pd
from sqlalchemy import create_engine, inspect

import config
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def _get_db_url(dbms, user, pwd, host, port, db):
    return f"{dbms}://{user}:{pwd}@{host}:{port}/{db}"


class SQLQuerier:
    """Class for performing SQL querying."""

    def __init__(self, config: argparse.Namespace):
        """Instantiate.

        Parameters
        ----------
        config: argparse.Namespace
            Configuration stored in object.
        engine: sqlalchemy.engine.base.Engine
            SQL extraction engine.
        inspector: sqlalchemy.engine.reflection.Inspector
            Module for schema inspection.
        query: str
            SQL query.
        data: pd.DataFrame
            Pandas dataframe container for storing extracted data.
        """
        self.config = config
        self.engine = create_engine(
            _get_db_url(
                config.dbms,
                config.user,
                config.password,
                config.host,
                config.port,
                config.database,
            )
        )
        self.inspector = inspect(self.engine)
        self.query = None
        self.data = None

    def load_query(self):
        """Load SQL query."""
        sql_file = open(self.config.sql_query_path)
        self.query = sql_file.read()

    def run_query(self) -> pd.DataFrame:
        """Run extraction.

        Returns
        -------
        pandas.DataFrame
            Extracted data as a pandas dataframe.
        """
        if not self.query:
            LOGGER.info("Loading SQL query...")
            self.load_query()
            LOGGER.info(f"SQL query file {self.config.sql_query_path} loaded!")
            LOGGER.info("Running query on DB...")
            start_time = time.time()
            self.data = pd.read_sql(self.query, con=self.engine)
            LOGGER.info(
                f"Extraction from SQL DB completed, in {time.time() - start_time} s!"
            )
            LOGGER.info(f"{self.data.count()} rows extracted!")
            return self.data
        else:
            LOGGER.info("Data extraction already done! Reset and try again!")

    def reset(self):
        """Reset query and data containers."""
        self.query = None
        self.data = None

    def print_sql_query(self):
        """Print SQL query."""
        if self.query is not None:
            LOGGER.info(self.query)


if __name__ == "__main__":
    cfg = config.read_config("../configs/default/*.yaml")
    extractor = SQLQuerier(cfg)
    data = extractor.run_query()
    print(data.count)
