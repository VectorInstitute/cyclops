"""Object Relational Mapper (ORM) using sqlalchemy."""

import time
import argparse
import logging

import pandas as pd

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import MetaData
from sqlalchemy.orm import sessionmaker

from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def _get_db_url(dbms: str, user: str, pwd: str, host: str, port: str, db: str) -> str:
    return f"{dbms}://{user}:{pwd}@{host}:{port}/{db}"


def _get_attr_name(name: str) -> str:
    return name[name.index(".") + 1 :]  # noqa: E203


class Schema:
    """Database schema wrapper.

    Attributes
    ----------
    name: str
        Name of schema.
    x: sqlalchemy.sql.schema.MetaData
        Metadata for schema.
    """

    def __init__(self, name: str, x: sqlalchemy.sql.schema.MetaData):
        """Instantiate.

        Parameters
        ----------
        name: str
            Name of schema.
        x: sqlalchemy.sql.schema.MetaData
            Metadata for schema.
        """
        self.name = name
        self.x = x


class Table:
    """Database table wrapper.

    Attributes
    ----------
    name: str
        Name of table.
    x: sqlalchemy.sql.schema.Table
        Metadata for schema.
    """

    def __init__(self, name: str, x: sqlalchemy.sql.schema.Table):
        """Instantiate.

        Parameters
        ----------
        name: str
            Name of table.
        x: sqlalchemy.sql.schema.Table
            Table schema.
        """
        self.name = name
        self.x = x


class DBMetaclass(type):
    """Meta class for Database, keeps track of instances for singleton."""

    __instances: dict = {}

    def __call__(cls, *args, **kwargs):
        """Call."""
        if cls not in cls.__instances:
            cls.__instances[cls] = super().__call__(*args, **kwargs)
        return cls.__instances[cls]


class Database(metaclass=DBMetaclass):
    """Database class (singleton).

    Attributes
    ----------
    config: argparse.Namespace
        Configuration stored in object.
    engine: sqlalchemy.engine.base.Engine
        SQL extraction engine.
    inspector: sqlalchemy.engine.reflection.Inspector
        Module for schema inspection.
    session: sqlalchemy.orm.session.Session
    """

    def __init__(self, config: argparse.Namespace):
        """Instantiate.

        Parameters
        ----------
        config: argparse.Namespace
            Configuration stored in object.
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

        # Create a session for using ORM.
        Session = sessionmaker(self.engine)
        Session.configure(bind=self.engine)
        self.session = Session()

        self._setup()
        LOGGER.info("Database setup, ready to run queries!")

    def _setup(self):
        """Prepare ORM DB."""
        meta = dict()
        # TODO: Unify this when using for mimic.
        # schemas = self.inspector.get_schema_names()

        for s in ["public"]:
            metadata = MetaData(schema=s)
            metadata.reflect(bind=self.engine)
            meta[s] = metadata
            schema = Schema(s, meta[s])
            for t in meta[s].tables:
                table = Table(t, meta[s].tables[t])
                for column in meta[s].tables[t].columns:
                    setattr(table, column.name, column)
                setattr(schema, _get_attr_name(table.name), table)
            setattr(self, s, schema)

    def run_query(self, query: sqlalchemy.sql.selectable.Subquery) -> pd.DataFrame:
        """Run query.

        Parameters
        ----------
        query: sqlalchemy.sql.selectable.Subquery

        Returns
        -------
        pd.DataFrame
            Extracted data from query.
        """
        start_time = time.time()
        with self.session.connection():
            data = pd.read_sql_query(query, self.engine)
        LOGGER.info(f"Query returned successfully, took {time.time() - start_time} s")
        return data
