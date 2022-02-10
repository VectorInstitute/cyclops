"""Object Relational Mapper (ORM) using sqlalchemy."""

import logging

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import MetaData

from sqlalchemy.orm import sessionmaker

import config
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def _get_db_url(dbms, user, pwd, host, port, db):
    return f"{dbms}://{user}:{pwd}@{host}:{port}/{db}"


def _get_attr_name(name: str) -> str:
    return name[name.index(".") + 1:]


class Schema:
    def __init__(self, name, x):
        self.name = name
        self.x = x


class Table:
    def __init__(self, name, x):
        self.name = name
        self.x = x


class DB:
    """Database class.

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

    def __init__(self, config):
        """Instantiate.

        Attributes
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

    def _setup(self):
        """Setup ORM DB."""
        meta = dict()
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

    def run_query(self, query):
        """Run query."""
        return pd.read_sql_query(query, self.engine)
