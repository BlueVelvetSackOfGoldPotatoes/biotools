async function request(path, payload = null, method = 'POST') {
  const options = { method, headers: {} }
  if (payload !== null) {
    options.headers['Content-Type'] = 'application/json'
    options.body = JSON.stringify(payload)
  }
  const response = await fetch(path, options)
  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.error || 'Request failed')
  }
  return data
}

export async function fetchPresets() {
  return request('/api/presets', null, 'GET')
}

export async function analyze(payload) {
  return request('/api/analyze', payload)
}

export async function simulate(payload) {
  return request('/api/simulate', payload)
}

export async function batch(payload) {
  return request('/api/batch', payload)
}
