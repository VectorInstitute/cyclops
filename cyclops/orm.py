"""Object Relational Mapper (ORM) using sqlalchemy."""

import logging
from collections import defaultdict

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import select, func, text, case
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import (
    Integer,
    Unicode,
    String,
    DateTime,
    Boolean,
    Numeric,
    Text,
    Date,
    UniqueConstraint,
    UnicodeText,
    Index,
)

from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Query

from sqlalchemy.sql import extract
from sqlalchemy.sql.expression import and_, or_, exists

import config
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def _get_db_url(dbms, user, pwd, host, port, db):
    return f"{dbms}://{user}:{pwd}@{host}:{port}/{db}"


def _get_attr_name(name: str) -> str:
    return name[name.index(".") + 1 :]


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
        meta = dict()
        schemas = self.inspector.get_schema_names()

        for s in ["public"]:
            metadata = MetaData(schema=s)
            metadata.reflect(bind=self.engine)
            meta[s] = metadata

            schema = Schema(s, meta[s])

            for t in meta[s].tables:
                table = Table(t, meta[s].tables[t])
                for c in meta[s].tables[t].columns:
                    # Set up column attributes in each table
                    setattr(table, c.name, c)
                # Set up table attributes in each schema
                setattr(schema, _get_attr_name(table.name), table)
            # Set up schema attributes in the database
            setattr(self, s, schema)


if __name__ == "__main__":
    cfg = config.read_config("../configs/default/*.yaml")
    db = DB(cfg)
    query = select(
        db.public.ip_administrative.patient_id_hashed.label("patient_id"),
        db.public.ip_administrative.genc_id,
        db.public.ip_administrative.hospital_id,
        case((db.public.ip_administrative.gender == "F", 1), else_=0).label("sex"),
        db.public.ip_administrative.age,
        case(
            (db.public.ip_administrative.discharge_disposition == 7, 1), else_=0
        ).label("mort_hosp"),
        db.public.ip_administrative.discharge_date_time,
        db.public.ip_administrative.admit_date_time,
        db.public.diagnosis.diagnosis_code.label("mr_diagnosis"),
        extract("year", db.public.ip_administrative.admit_date_time).label("year"),
    )
    df = pd.read_sql_query(query, db.engine)
    print(df.count())
