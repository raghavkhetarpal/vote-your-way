import axios from 'axios'

const api = axios.create({
  // Leave unset for local Vite proxying; set VITE_API_BASE_URL to the Render
  // service URL (without a trailing slash) in Vercel.
  baseURL: `${import.meta.env.VITE_API_BASE_URL || ''}/api`,
  timeout: 30000,
})

// Store auth credentials
let authCredentials = null

export const setAuthCredentials = (username, password) => {
  if (username && password) {
    authCredentials = `${username}:${password}`
    // Update default header for all requests
    api.defaults.headers.common['Authorization'] = `Basic ${btoa(authCredentials)}`
  } else {
    authCredentials = null
    delete api.defaults.headers.common['Authorization']
  }
}

export const getStatus = () => api.get('/status')
export const runPipeline = (forceRerun = false, useScraper = false) =>
  api.post('/pipeline/run', { force_rerun: forceRerun, use_scraper: useScraper })

export const getManifestoes = () => api.get('/manifestoes')
export const getManifestoText = (party, year) => api.get(`/manifestoes/${party}/${year}/text`)

export const getPromises = (params = {}) => api.get('/promises', { params })
export const getScores = () => api.get('/scores')
export const getCustomScores = (data) => api.post('/scores/custom', data)
export const getRecommendation = (priorityCategory = null) =>
  api.get('/recommendation', { params: priorityCategory ? { priority_category: priorityCategory } : {} })

export const getClustering = () => api.get('/clustering')
export const getApriori = () => api.get('/apriori')
export const getOverview = () => api.get('/analytics/overview')
export const getCategoryAnalysis = (category) => api.get(`/analytics/category/${category}`)

export default api
