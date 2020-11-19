import api from './request';

const urls = {
  root: '/xmldrugs'
};

function fetch() {
  return api.get(urls.root);
}

export default { fetch };
