import uniqBy from 'lodash/uniqBy';

const relevantProps = [
  'logP',
  'logS',
  'Water Solubility',
  'Molecular Weight',
  'Polar Surface Area (PSA)',
  'Refractivity'
];

function parseJsonProps(drugs) {
  return drugs.map(drug => {
    const cprops = JSON.parse(drug.cprops)['calculated-properties'];
    return { ...drug, cprops: pruneProps(cprops) };
  });
}

function pruneProps(props) {
  if (!props) return;
  const uniqProps = uniqBy(props.property, 'kind');
  return uniqProps.filter(({ kind }) => relevantProps.includes(kind));
}

export default parseJsonProps;
