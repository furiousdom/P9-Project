from lxml import etree

from config import xml_datasets as xdata

def parse(dataset, parser):
    """
    Loads the specified XML and parses it for either SMILES or FASTA strings.
    """
    print('Loading XML...')
    root = etree.parse(xdata[dataset]).getroot()
    print('Loading completed.')

    if parser == 'SMILES':
        print('SMILES Parser start...')
        result = parseSmiles(root)
        print('Parser finished.')
        return result
    elif parser == 'FASTA':
        print('FASTA Parser start...')
        result = parseFasta(root)
        print('Parser finished.')
        return result

def parseSmiles(root):
    """
    Parses the loaded XML for SMILES strings. Returns a dictionary of
    molecules with keys as drug names and SMILES strings as values.
    """
    small_molecules = {}

    for child in root:
        name = child.find(xdata['ns'] + 'name')
        id = child.find(xdata['ns'] + 'drugbank-id')
        atts = child.attrib
        if atts['type'] == 'small molecule':
            properties = child.find(xdata['ns'] + 'calculated-properties')
            for prop in properties:
                kind = prop.find(xdata['ns'] + 'kind')
                value = prop.find(xdata['ns'] + 'value')
                if kind.text == 'SMILES':
                    small_molecules[id.text] = value.text
    return small_molecules

def parseFasta(root):
    proteins = {}

    for child in root:
        # name = child.find(xdata['ns'] + 'name')
        id = child.find(xdata['ns'] + 'drugbank-id')
        atts = child.attrib
        if atts['type'] == 'biotech':
            sequences = child.find(xdata['ns'] + 'sequences')
            for sequence in sequences:
                if sequence.attrib['format'] == 'FASTA':
                    proteins[id.text] = sequence.text
    return proteins

def readProteinIds():
    ids = []
    f = open('./data/drugbank_IDs.txt', 'r', encoding="utf-8")
    for line in f.readlines():
        ids.append(line.strip())
    f.close()
    return ids

def parseExtIds(dataset):
    """
    Loads the specified XML and parses it for CIDs and Uniprot IDs.
    """
    print('Loading XML...')
    root = etree.parse(xdata[dataset]).getroot()
    print('Loading completed.')
    print('Parsing started...')
    cids = parseCID(root)
    uniprots = parseUniprotIds(root)
    print('Parsing completed.')
    return cids, uniprots

def parseCID(root):
    small_molecules = {}

    for child in root:
        id = child.find(xdata['ns'] + 'drugbank-id')
        atts = child.attrib
        if atts['type'] == 'small molecule':
            externalIdentifiers = child.find(xdata['ns'] + 'external-identifiers')
            for extIdentifier in externalIdentifiers:
                resource = extIdentifier.find(xdata['ns'] + 'resource')
                identifier = extIdentifier.find(xdata['ns'] + 'identifier')
                if resource.text == 'PubChem Compound':
                    small_molecules[id.text] = identifier.text
    return small_molecules

def parseUniprotIds(root):
    small_molecules = {}

    for child in root:
        id = child.find(xdata['ns'] + 'drugbank-id')
        atts = child.attrib
        if atts['type'] == 'biotech':
            externalIdentifiers = child.find(xdata['ns'] + 'external-identifiers')
            for extIdentifier in externalIdentifiers:
                resource = extIdentifier.find(xdata['ns'] + 'resource')
                identifier = extIdentifier.find(xdata['ns'] + 'identifier')
                if resource.text == 'UniProtKB':
                    small_molecules[id.text] = identifier.text
    return small_molecules
