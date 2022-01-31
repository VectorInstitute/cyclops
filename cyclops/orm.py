"""Figure out ORMs."""


from collections import defaultdict

from sqlalchemy import select, case
from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.ext.automap import automap_base


Base = automap_base()
CONN_STR = "postgresql://krishnanam:Totem110#@db.gemini-hpc.ca:5432/delirium_v3_0_0"
engine = create_engine(CONN_STR, echo=True)
session = Session(engine, future=True)

Base.prepare(engine, reflect=True)

table_names = ["ip_administrative", "er_administrative", "diagnosis"]

tables = defaultdict(None)
for table_name in table_names:
    tables[table_name] = getattr(Base.classes, table_name)


# query with ORM columns
statement = select(
    getattr(tables["ip_administrative"], "patient_id_hashed"),
    getattr(tables["ip_administrative"], "genc_id"),
    getattr(tables["ip_administrative"], "hospital_id"),
    case((getattr(tables["ip_administrative"], "gender") == "F", 1), else_=0),
    getattr(tables["ip_administrative"], "age"),
    case((getattr(tables["ip_administrative"], "discharge_disposition") == 7, 1), else_=0),
)

# list of tuples
result = session.execute(statement).first()

print(len(result), result)
