from lxml import etree

from config import xml_datasets as xdata

def parse(dataset, parser):
    """
    Loads the specified XML and parses it for either SMILES or FASTA strings,
    depending on the specified parser.
    """
    print('Loading XML...')
    root = etree.parse(xdata[dataset]).getroot()
    print('Loading completed.')

    if parser == 'SMILES':
        print('SMILES Parser start...')
        result = parse_smiles(root)
        print('Parser finished.')
        return result
    elif parser == 'FASTA':
        print('FASTA Parser start...')
        result = parse_fasta(root)
        print('Parser finished.')
        return result


def parse_smiles_and_fasta(dataset):
    """
    Loads the specified XML and parses it for either SMILES or FASTA strings.
    """
    print('Loading XML...')
    root = etree.parse(xdata[dataset]).getroot()
    print('Loading completed.')

    print('SMILES Parser start...')
    smiles = parse_smiles(root)
    print('Parser finished.')
    print('FASTA Parser start...')
    fasta = parse_fasta(root)
    print('Parser finished.')
    return smiles, fasta

def parse_smiles(root):
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

def parse_fasta(root):
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

def read_protein_ids():
    ids = []
    drugbank_ids_file = open('./data/drugbank_ids.txt', 'r', encoding="utf-8")
    for line in drugbank_ids_file.readlines():
        ids.append(line.strip())
    drugbank_ids_file.close()
    return ids

def parse_ext_ids(dataset):
    """
    Loads the specified XML and parses it for CIDs and Uniprot IDs.
    """
    print('Loading XML...')
    root = etree.parse(xdata[dataset]).getroot()
    print('Loading completed.')
    print('Parsing started...')
    cids = parse_cid(root)
    uniprots = parse_uniprot_ids(root)
    print('Parsing completed.')
    return cids, uniprots

def parse_cid(root):
    small_molecules = {}

    for child in root:
        id = child.find(xdata['ns'] + 'drugbank-id')
        atts = child.attrib
        if atts['type'] == 'small molecule':
            external_identifiers = child.find(xdata['ns'] + 'external-identifiers')
            for ext_identifier in external_identifiers:
                resource = ext_identifier.find(xdata['ns'] + 'resource')
                identifier = ext_identifier.find(xdata['ns'] + 'identifier')
                if resource.text == 'PubChem Compound':
                    small_molecules[id.text] = identifier.text
    return small_molecules

def parse_uniprot_ids(root):
    small_molecules = {}

    for child in root:
        id = child.find(xdata['ns'] + 'drugbank-id')
        atts = child.attrib
        if atts['type'] == 'biotech':
            external_identifiers = child.find(xdata['ns'] + 'external-identifiers')
            for ext_identifier in external_identifiers:
                resource = ext_identifier.find(xdata['ns'] + 'resource')
                identifier = ext_identifier.find(xdata['ns'] + 'identifier')
                if resource.text == 'UniProtKB':
                    small_molecules[id.text] = identifier.text
    return small_molecules
