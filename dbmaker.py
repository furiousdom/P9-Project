# XML Parser Library
from lxml import etree

import psycopg2

#XML Namespace
xmlns = "{http://www.drugbank.ca}"

def log2q(s):
    with open("q", "a", encoding="utf-8") as f:
        f.write(s)
        f.write("\n\n")
        f.close()

def exc(e):
    if hasattr(e, 'message'):
        print(e.message)
    else:
        print(e)

def getText(node, arg):
    try:
        return node.find(xmlns + arg).text
    except Exception as e:
        exc(e)
        return ""

def getDate(node, arg):
    try:
        txt = str(node.find(xmlns + arg).text)
        #print(txt)
        if txt.find("None"):
            return txt
        else:
            return "0001-01-01"
    except Exception as e:
        exc(e)
        return "0001-01-01"

def getXML(node, arg):
    try:
        tmp = str(etree.tostring(node.find(xmlns + arg)).decode('UTF-8'))
        tmp = tmp.replace("\n", "").strip()
        return tmp
    except Exception as e:
        exc(e)
        return ""
        # return f'<{arg}/>' Uncomment only when creating Calculated Properties Table

# Parse into ElementTree tree
print("Starting Parsing")

#Full Set
tree = etree.parse("data/full_database.xml")

#Test Set
#tree = etree.parse("drugbank_record.xml")

CONN_STRING = """host='localhost' port='5432' dbname='drugdb' user='postgres' password='postgres' sslmode='disable'"""

# Open a DB session

dbSession = psycopg2.connect(CONN_STRING)

# Open a database cursor

dbCursor = dbSession.cursor()

# Identify Root of the Tree
root = tree.getroot()
print(root.tag)

#MAIN TABLE CREATION FUNCTION
def createMainTable():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        #MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        name = getText(child, "name")
        description = getText(child, "description")
        cas_number = getText(child, "cas-number")
        unii = getText(child, "unii")
        state = getText(child, "state")
        indication = getText(child, "indication")
        pharmacodynamics = getText(child, "pharmacodynamics")
        mechanism = getText(child, "mechanism-of-action")
        toxicity = getText(child, "toxicity")
        metabolism = getText(child, "metabolism")
        absorbtion = getText(child, "absorption")
        halflife = getText(child, "half-life")
        protein_binding = getText(child, "protein-binding")
        route_of_elimination = getText(child, "route-of-elimination")
        volume_of_distribution = getText(child, "volume-of-distribution")
        clearance = getText(child, "clearance")
        classification = getXML(child, "classification")
        fda_label = getText(child, "fda-label")
        msds = getText(child, "msds")
        reactions = getXML(child, "reactions")
        snp_effects = getXML(child, "snp-effects")

        print(iterator, ": ", nametech.text)

        sqlInsertRow1 = f"INSERT INTO main_table values($${primary_id}$$,$${name}$$,$${description}$$,$${cas_number}$$,$${unii}$$,$${state}$$,$${indication}$$,$${pharmacodynamics}$$,$${mechanism}$$,$${toxicity}$$,$${metabolism}$$,$${absorbtion}$$,$${halflife}$$,$${protein_binding}$$,$${route_of_elimination}$$,$${volume_of_distribution}$$,$${clearance}$$,$${classification}$$,$${fda_label}$$,$${msds}$$,$${reactions}$$,$${snp_effects}$$);"
        log2q(sqlInsertRow1)
        try:
            dbCursor.execute(sqlInsertRow1)
            dbSession.commit()
        except Exception as e:
            errors += 1
            print(f"Error in {name} insertion.{exc(e)}")

        #if iterator == 10:
        #    break
        iterator += 1

    print(f"We have encountered {errors} errors during the upload.")

# SECONDARY IDS TABLE CREATION FUNCTION
def createSecondaryIDs():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        ids = child.findall(xmlns+"drugbank-id")
        for id in ids:
            if id.text == primary_id:
                continue
            else:
                sqlInsertRow1 = f"INSERT INTO secondary_ids_table values($${iterator}$$,$${primary_id}$$,$${id.text}$$)"
                log2q(sqlInsertRow1)
                try:
                   dbCursor.execute(sqlInsertRow1)
                   dbSession.commit()
                   iterator += 1
                except Exception as e:
                   errors += 1
                   print(f"Error in {nametech} insertion.{exc(e)}")
        #print(ids)
    print(f"We have encountered {errors} errors during the upload.")

# CREATE GROUPS
def createGroups():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        groupsNode = child.find(xmlns+"groups")
        groups = groupsNode.findall(xmlns + "group")
        for group in groups:
            sqlInsertRow1 = f"INSERT INTO groups_table values($${iterator}$$,$${primary_id}$$,$${group.text}$$)"
            log2q(sqlInsertRow1)
            try:
                dbCursor.execute(sqlInsertRow1)
                dbSession.commit()
                iterator += 1
            except Exception as e:
                errors += 1
                print(f"Error in {nametech} insertion.{exc(e)}")
        #print(groups)
    print(f"We have encountered {errors} errors during the upload.")

# CREATE CATEGORIES
def createCategories():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        categoriesNode = child.find(xmlns+"categories")
        categories = categoriesNode.findall(xmlns + "category")
        for category in categories:
            cat = getText(category, "category")
            mesh_id = getText(category, "mesh-id")

            sqlInsertRow1 = f"INSERT INTO categories_table values($${iterator}$$,$${primary_id}$$,$${cat}$$,$${mesh_id}$$)"
            log2q(sqlInsertRow1)
            try:
                dbCursor.execute(sqlInsertRow1)
                dbSession.commit()
                iterator += 1
            except Exception as e:
                errors += 1
                print(f"Error in {nametech} insertion.{exc(e)}")
        #print(groups)
    print(f"We have encountered {errors} errors during the upload.")

# CREATE DOSAGE
def createDosages():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        dosagesNode = child.find(xmlns+"dosages")
        dosages = dosagesNode.findall(xmlns + "dosage")
        for dosage in dosages:
            form = getText(dosage, "form")
            route = getText(dosage, "route")
            strength = getText(dosage, "strength")
            sqlInsertRow1 = f"INSERT INTO dosages_table values($${iterator}$$,$${primary_id}$$,$${form}$$,$${route}$$,$${strength}$$)"
            log2q(sqlInsertRow1)
            try:
                dbCursor.execute(sqlInsertRow1)
                dbSession.commit()
                iterator += 1
            except Exception as e:
                errors += 1
                print(f"Error in {nametech} insertion.{exc(e)}")
        #print(groups)
    print(f"We have encountered {errors} errors during the upload.")

# CREATE SYNONYMS
def createSynonyms():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        synonymsNode = child.find(xmlns+"synonyms")
        synonyms = synonymsNode.findall(xmlns + "synonym")
        for synonym in synonyms:
            sqlInsertRow1 = f"INSERT INTO synonyms_table values($${iterator}$$,$${primary_id}$$,$${synonym.text}$$)"
            log2q(sqlInsertRow1)
            try:
                dbCursor.execute(sqlInsertRow1)
                dbSession.commit()
                iterator += 1
            except Exception as e:
                errors += 1
                print(f"Error in {nametech} insertion.{exc(e)}")
        #print(groups)
    print(f"We have encountered {errors} errors during the upload.")

# CREATE PRODUCTS
def createProducts():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        productsNode = child.find(xmlns+"products")
        products = productsNode.findall(xmlns + "product")
        for product in products:
            product_name = getText(product, "name")
            labeller = getText(product, "labeller")
            ndc_id = getText(product, "ndc-id")
            ndc_product_code = getText(product, "ndc-product-code")
            dpd_id =  getText(product, "dpd-id")
            ema_product_code = getText(product, "ema-product-code")
            ema_ma_number = getText(product, "ema-ma-number")
            started_marketing_on = getDate(product, "started-marketing-on")
            ended_marketing_on = getDate(product, "ended-marketing-on")
            dosage_form = getText(product, "dosage-form")
            strength = getText(product, "strength")
            route = getText(product, "route")
            fda_application_number = getText(product, "fda-application-number")
            generic = getText(product, "generic")
            over_the_counter = getText(product, "over-the-counter")
            approved = getText(product, "approved")
            country = getText(product, "country")
            source = getText(product, "source")

            sqlInsertRow1 = f"INSERT INTO products_table values($${iterator}$$,$${primary_id}$$,$${product_name}$$,$${labeller}$$,$${ndc_id}$$,$${ndc_product_code}$$,$${dpd_id}$$,$${ema_product_code}$$,$${ema_ma_number}$$,$${started_marketing_on}$$,$${ended_marketing_on}$$,$${dosage_form}$$,$${strength}$$,$${route}$$,$${fda_application_number}$$,$${generic}$$,$${over_the_counter}$$,$${approved}$$,$${country}$$,$${source}$$)"
            log2q(sqlInsertRow1)
            try:
                dbCursor.execute(sqlInsertRow1)
                dbSession.commit()
                iterator += 1
            except Exception as e:
                errors += 1
                print(f"Error in {nametech} insertion.{exc(e)}")
        #print(groups)
    print(f"We have encountered {errors} errors during the upload.")

# CREATE MIXTURES
def createMixtures():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        mixturesNode = child.find(xmlns+"mixtures")
        mixtures = mixturesNode.findall(xmlns + "mixture")
        for mixture in mixtures:
            mixture_name = getText(mixture, "name")
            ingredient = getText(mixture, "ingredients")

            sqlInsertRow1 = f"INSERT INTO mixtures_table values($${iterator}$$,$${primary_id}$$,$${mixture_name}$$,$${ingredient}$$)"
            log2q(sqlInsertRow1)
            try:
                dbCursor.execute(sqlInsertRow1)
                dbSession.commit()
                iterator += 1
            except Exception as e:
                errors += 1
                print(f"Error in {nametech} insertion.{exc(e)}")
        #print(groups)
    print(f"We have encountered {errors} errors during the upload.")

# CREATE ATC CODES
def createATCcodes():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        atc_codes = getXML(child, "atc-codes")

        sqlInsertRow1 = f"INSERT INTO atc_codes_table values($${iterator}$$,$${primary_id}$$,$${atc_codes}$$)"
        log2q(sqlInsertRow1)
        try:
            dbCursor.execute(sqlInsertRow1)
            dbSession.commit()
            iterator += 1
        except Exception as e:
            errors += 1
            print(f"Error in {nametech} insertion.{exc(e)}")

    print(f"We have encountered {errors} errors during the upload.")

# CREATE INTERACTIONS
# ??? ALL OF INTERACTIONS ARE THERE? WERE SOME SKIPPED?!?! I THINK NOT.
def createInteractions():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        interactionsNode = child.find(xmlns+"drug-interactions")
        interactions = interactionsNode.findall(xmlns + "drug-interaction")
        for interaction in interactions:
            drug_id_2 = getText(interaction, "drugbank-id")
            sd_name = getText(interaction, "name")
            sd_desc = getText(interaction, "description")

            sqlInsertRow1 = f"INSERT INTO drug_interactions_table values($${iterator}$$,$${primary_id}$$,$${drug_id_2}$$,$${sd_name}$$,$${sd_desc}$$)"
            log2q(sqlInsertRow1)
            try:
                dbCursor.execute(sqlInsertRow1)
                dbSession.commit()
                iterator += 1
            except Exception as e:
                errors += 1
                print(f"Error in {nametech} insertion.{exc(e)}")
        #print(groups)
    print(f"We have encountered {errors} errors during the upload.")

# CREATE SEQUENCES
def createSequences():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        if child.find(xmlns+"sequences") is not None:
            sequencesNode = child.find(xmlns+"sequences")
            sequences = sequencesNode.findall(xmlns+"sequence")
            for sequence in sequences:
                sequence_t = sequence.text
                sqlInsertRow1 = f"INSERT INTO sequences_table values($${iterator}$$,$${primary_id}$$,$${sequence_t}$$)"
                log2q(sqlInsertRow1)
                try:
                    dbCursor.execute(sqlInsertRow1)
                    dbSession.commit()
                    iterator += 1
                except Exception as e:
                    errors += 1
                    print(f"Error in {nametech} insertion.{exc(e)}")
    print(f"We have encountered {errors} errors during the upload.")

# CREATE Properties
def createProperties():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        properties = getXML(child, "experimental-properties")

        sqlInsertRow1 = f"INSERT INTO properties_table values($${iterator}$$,$${primary_id}$$,$${properties}$$)"
        log2q(sqlInsertRow1)
        try:
            dbCursor.execute(sqlInsertRow1)
            dbSession.commit()
            iterator += 1
        except Exception as e:
            errors += 1
            print(f"Error in {nametech} insertion.{exc(e)}")

    print(f"We have encountered {errors} errors during the upload.")

# CREATE Calculated Properties
def createCalcProperties():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        properties = getXML(child, "calculated-properties")

        sqlInsertRow1 = f"INSERT INTO calc_properties_table values($${iterator}$$,$${primary_id}$$,$${properties}$$)"
        log2q(sqlInsertRow1)
        try:
            dbCursor.execute(sqlInsertRow1)
            dbSession.commit()
            iterator += 1
        except Exception as e:
            errors += 1
            print(f"Error in {nametech} insertion.{exc(e)}")

    print(f"We have encountered {errors} errors during the upload.")

# CREATE PATHWAYS
def createPathways():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        pathways = getXML(child, "pathways")

        sqlInsertRow1 = f"INSERT INTO pathways_table values($${iterator}$$,$${primary_id}$$,$${pathways}$$)"
        log2q(sqlInsertRow1)
        try:
            dbCursor.execute(sqlInsertRow1)
            dbSession.commit()
            iterator += 1
        except Exception as e:
            errors += 1
            print(f"Error in {nametech} insertion.{exc(e)}")

    print(f"We have encountered {errors} errors during the upload.")

# CREATE TARGETS
def createTargets():
    # Iterate over all children
    iterator = 1
    errors = 0
    for child in root:
        nametech = child.find(xmlns + "name")

        # MAIN TABLE INSERT
        primary_id = getText(child, "drugbank-id")
        targets = getXML(child, "targets")

        sqlInsertRow1 = f"INSERT INTO targets_table values($${iterator}$$,$${primary_id}$$,$${targets}$$)"
        log2q(sqlInsertRow1)
        try:
            dbCursor.execute(sqlInsertRow1)
            dbSession.commit()
            iterator += 1
        except Exception as e:
            errors += 1
            print(f"Error in {nametech} insertion.{exc(e)}")

    print(f"We have encountered {errors} errors during the upload.")


# MAIN SECTION
# print('Creating main table...')
# createMainTable()
# print('Finished creating main table...')

# print('Creating SecondaryIds table...')
# createSecondaryIDs()
# print('Finished creating SecondaryIds table...')

# print('Creating Groups table...')
# createGroups()
# print('Finished creating Groups table...')

# print('Creating Categories table...')
# createCategories()
# print('Finished creating Categories table...')

# print('Creating Dosages table...')
# createDosages()
# print('Finished creating Dosages table...')

# print('Creating Synonyms table...')
# createSynonyms()
# print('Finished creating Synonyms table...')

# print('Creating Products table...')
# createProducts()
# print('Finished creating Products table...')

# print('Creating Mixtures table...')
# createMixtures()
# print('Finished creating Mixtures table...')

# print('Creating ATCcodes table...')
# createATCcodes()
# print('Finished creating ATCcodes table...')

# print('Creating Interactions table...')
# createInteractions()
# print('Finished creating Interactions table...')

# print('Creating Sequences table...')
# createSequences()
# print('Finished creating Sequences table...')

# print('Creating Properties table...')
# createProperties()
# print('Finished creating Properties table...')

# print('Creating Pathways table...')
# createPathways()
# print('Finished creating Pathways table...')

# print('Creating Targets table...')
# createTargets()
# print('Finished creating Targets table...')

# TODO: Before running:
# modify the exception in getXML function,
# comment the return of empty string and uncomment the line below it.
# When done return the function the way it was!

# print('Creating Calculates Properties table...')
# createCalcProperties()
# print('Finished creating Calculates Properties table...')

print("Dunzo")
