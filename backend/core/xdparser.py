from lxml import etree

from config import xml_datasets as xdata

def aces():
    ns = 'ns'
    small_molecule = 'small molecule'
    calculated_properties = 'calculated-properties'
    kind = 'kind'
    value = 'value'
    smiles = 'SMILES'
    return ns, small_molecule, calculated_properties, kind, value, smiles

def parseSmiles(dataset):
    """
    Loads the specified XML and parses it for SMILES strings. Returns a
    dictionary of molecules with keys as drug names and SMILES strings as values.
    """
    print('Loading XML...')
    root = etree.parse(xdata[dataset]).getroot()
    small_molecules = {}
    ns, sm, cp, akind, avalue, asmiles = aces()

    print('Loading copleted. Parser start...')

    for child in root:
        name = child.find(xdata[ns] + 'name')
        atts = child.attrib
        if atts['type'] == sm:
            properties = child.find(xdata[ns] + cp)
            for prop in properties:
                kind = prop.find(xdata[ns] + akind)
                value = prop.find(xdata[ns] + avalue)
                if kind.text == asmiles:
                    small_molecules[name.text] = value.text

    print('Parser finished.')
    return small_molecules

def parseFasta():
    return 'Not implemented yet'
