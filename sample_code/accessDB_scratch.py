import pyodbc
import getpass
import pandas as pd

# Use your own username and whichever database you need
server = 'db.gemini-hpc.ca' 
database = 'delirium_v3_0_0' 
username = 'koshkinam'
password = getpass.getpass(prompt='Database password: ', stream=None) 
con = pyodbc.connect('DRIVER={PostgreSQL};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+ password)
ip_adm_df = pd.read_sql_query('SELECT * FROM diagnosis LIMIT 10',con=con)
print(ip_adm_df.head())

