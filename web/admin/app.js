const rawApiBase = document.body.dataset.apiBase || '';
const apiBase = (window.location.hostname === '127.0.0.1' && rawApiBase.includes('//localhost:'))
  ? rawApiBase.replace('//localhost:', '//127.0.0.1:')
  : (window.location.hostname === 'localhost' && rawApiBase.includes('//127.0.0.1:'))
  ? rawApiBase.replace('//127.0.0.1:', '//localhost:')
  : rawApiBase;
const token = document.querySelector('#token'); const files = document.querySelector('#files'); const status = document.querySelector('#status');
const documents = document.querySelector('#documents'); const usage = document.querySelector('#usage');
const headers = () => {
  const raw = (token.value || '').trim();
  const cleanToken = raw.replace(/^Bearer\s+/i, '');
  return cleanToken ? { Authorization: `Bearer ${cleanToken}` } : {};
};
function report(message, error = false) { status.textContent = message; status.className = error ? 'error' : ''; }
async function refresh() {
  try {
    const response = await fetch(`${apiBase}/api/documents`, { headers: headers() }); const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Could not load documents.');
    usage.className = 'usage'; usage.textContent = `${payload.usage.file_count} original PDFs · ${payload.usage.bytes_used.toLocaleString()} bytes used${payload.usage.estimated_remaining_bytes == null ? '' : ` · ${payload.usage.estimated_remaining_bytes.toLocaleString()} bytes remaining`}`;
    documents.replaceChildren();
    payload.documents.forEach((doc) => { const item = document.createElement('article'); item.className = 'document'; const strong = document.createElement('strong'); strong.textContent = doc.filename; const small = document.createElement('small'); small.textContent = `${doc.state} · ${doc.chunk_count} chunks · ${doc.content_hash}`; item.append(strong, small); documents.append(item); });
  } catch (error) { report(error.message, true); }
}
document.querySelector('#upload').addEventListener('click', async () => {
  if (!token.value || !files.files.length) { report('Provide the admin token and at least one PDF.', true); return; }
  const body = new FormData(); Array.from(files.files).forEach((file) => body.append('files', file)); report('Validating and indexing…');
  try { const response = await fetch(`${apiBase}/api/uploads`, { method: 'POST', headers: headers(), body }); const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || 'Upload failed.'); report(payload.map((job) => `${job.filename}: ${job.state}`).join(' · ')); await refresh(); } catch (error) { report(error.message, true); }
});
document.querySelector('#refresh').addEventListener('click', refresh);
