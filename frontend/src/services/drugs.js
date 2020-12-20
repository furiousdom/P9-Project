import api from './request';

const urls = {
  root: '/drugs'
};

function fetch() {
  return api.get(urls.root);
}

export default { fetch };
