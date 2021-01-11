import api from './request';

const urls = {
  root: '/drugs',
  search: '/search/'
};

function fetch() {
  return api.get(urls.root);
}

function search(params) {
  return api.post(`${urls.root}${urls.search}`, params);
}

export default { fetch, search };
