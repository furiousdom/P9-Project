import axios from 'axios';

const config = {
  headers: { 'Content-Type': 'application/json' },
  baseURL: 'http://127.0.0.1:8000/api'
};
const axiosInstance = axios.create(config);

export default axiosInstance;
