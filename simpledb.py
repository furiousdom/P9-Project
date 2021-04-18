# XML Parser Library
from lxml import etree

import psycopg2

#XML Namespace
xmlns = "{http://www.drugbank.ca}"

# Parse into ElementTree tree
print("Starting Parsing")

#Full Set
tree = etree.parse("db.xml")

#Test Set
#tree = etree.parse("drugbank_record.xml")

CONN_STRING = """host='localhost' port='5434' dbname='xmlDB' user='postgres' password='4523'"""

# Open a DB session

db_session = psycopg2.connect(host='localhost', port='5434', dbname='xmlDB', password='4523', user='postgres')

# Open a database cursor

db_cursor = db_session.cursor()

sqlCreateTable = "CREATE TABLE drugTable(id bigint, name text, content xml);"
db_cursor.execute(sqlCreateTable)
db_session.commit()

# Identify Root of the Tree
root = tree.getroot()
print(root.tag)
# Iterate over all children
iterator = 1
errors = 0
for child in root:
    name = child.find(xmlns + "name")
    print(iterator, ": ", name.text)
    #print(etree.tostring(child))
    xml_str = str(etree.tostring(child))
    xml_name = str(name.text)

    xml_str = xml_str[2:-1].replace("'", "''")
    xml_name = xml_name.replace("'","''")

    xml_str = xml_str.replace('\n', '')
    xml_name = xml_name.replace('\n', '')
    sql_insert_row = "INSERT INTO drugtable values("+str(iterator)+", '"+xml_name+"', '"+xml_str[:-2]+"');"
    iterator += 1
    try:
        db_cursor.execute(sql_insert_row)
        db_session.commit()
    except:
        errors += 1

print(f"We have encountered {errors} errors during the upload.")
print("Dunzo")
