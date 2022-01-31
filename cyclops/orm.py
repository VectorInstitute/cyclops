"""Figure out ORMs."""

import logging
from collections import defaultdict

from sqlalchemy import select, case
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base

import config
from cyclops.utils.log import setup_logging, LOG_FILE_PATH


# Logging.
LOGGER = logging.getLogger(__name__)
setup_logging(log_path=LOG_FILE_PATH, print_level="INFO", logger=LOGGER)


def _get_db_url(dbms, user, pwd, host, port, db):
    return f"{dbms}://{user}:{pwd}@{host}:{port}/{db}"


cfg = config.read_config("../configs/default/*.yaml")

Base = automap_base()
CONN_STR = _get_db_url(
    cfg.dbms,
    cfg.user,
    cfg.password,
    cfg.host,
    cfg.port,
    cfg.database,
)
engine = create_engine(CONN_STR, echo=True)
session = Session(engine, future=True)

Base.prepare(engine, reflect=True)

table_names = ["ip_administrative", "er_administrative", "diagnosis"]

tables = defaultdict(None)  # type: ignore
for table_name in table_names:
    tables[table_name] = getattr(Base.classes, table_name)


# query with ORM columns
statement = select(
    getattr(tables["ip_administrative"], "patient_id_hashed"),
    getattr(tables["ip_administrative"], "genc_id"),
    getattr(tables["ip_administrative"], "hospital_id"),
    case((getattr(tables["ip_administrative"], "gender") == "F", 1), else_=0),
    getattr(tables["ip_administrative"], "age"),
    case(
        (getattr(tables["ip_administrative"], "discharge_disposition") == 7, 1), else_=0
    ),
)

# list of tuples
result = session.execute(statement).first()
print(len(result), result)
