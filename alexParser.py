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

dbSession = psycopg2.connect(host='localhost', port='5434', dbname='xmlDB', password='4523', user='postgres')

# Open a database cursor

dbCursor = dbSession.cursor()

sqlCreateTable = "CREATE TABLE drugTable(id bigint, name text, content xml);"
dbCursor.execute(sqlCreateTable)
dbSession.commit()

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
    xmlstr = str(etree.tostring(child))
    xmlname = str(name.text)

    xmlstr = xmlstr[2:-1].replace("'", "''")
    xmlname = xmlname.replace("'","''")

    xmlstr = xmlstr.replace('\n', '')
    xmlname = xmlname.replace('\n', '')
    sqlInsertRow1 = "INSERT INTO drugtable values("+str(iterator)+", '"+xmlname+"', '"+xmlstr[:-2]+"');"
    iterator += 1
    try:
        dbCursor.execute(sqlInsertRow1)
        dbSession.commit()
    except:
        errors += 1

print(f"We have encountered {errors} errors during the upload.")
print("Dunzo")