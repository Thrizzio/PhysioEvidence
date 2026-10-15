const form = document.querySelector('#question-form');
const question = document.querySelector('#question');
const status = document.querySelector('#status');
const answer = document.querySelector('#answer');
const answerText = document.querySelector('#answer-text');
const evidenceNote = document.querySelector('#evidence-note');
const sources = document.querySelector('#sources');
const rawApiBase = document.body.dataset.apiBase || '';
const apiBase = (window.location.hostname === '127.0.0.1' && rawApiBase.includes('//localhost:'))
  ? rawApiBase.replace('//localhost:', '//127.0.0.1:')
  : (window.location.hostname === 'localhost' && rawApiBase.includes('//127.0.0.1:'))
  ? rawApiBase.replace('//127.0.0.1:', '//localhost:')
  : rawApiBase;

document.querySelectorAll('[data-example]').forEach((button) => {
  button.addEventListener('click', () => { question.value = button.dataset.example; question.focus(); });
});

function text(element, value) { element.textContent = value || ''; }

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  text(status, 'Searching the evidence corpus…');
  status.className = 'status';
  answer.hidden = true;
  try {
    const response = await fetch(`${apiBase}/api/query`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question.value.trim(), hybrid: document.querySelector('#hybrid').checked }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'The search could not be completed.');
    const result = payload.answer;
    text(status, '');
    text(answerText, result.answer);
    text(evidenceNote, result.insufficient_evidence ? 'The corpus did not contain enough relevant evidence for this question.' : `Strategy: ${payload.retrieval_strategy}. References below are actual retrieved passages.`);
    sources.replaceChildren();
    (result.sources || []).forEach((source) => {
      const card = document.createElement('article'); card.className = 'source';
      const title = document.createElement('h3'); title.textContent = `[Source ${source.source_number}] ${source.title || 'Untitled document'}`;
      const meta = document.createElement('p'); meta.className = 'source-meta';
      const details = [source.authors?.join(', '), source.publication_year, source.journal, source.section, source.pages, source.doi].filter(Boolean);
      meta.textContent = details.join(' · ');
      const excerpt = document.createElement('p'); excerpt.className = 'excerpt'; excerpt.textContent = source.excerpt || 'No excerpt was returned.';
      card.append(title, meta, excerpt); sources.append(card);
    });
    answer.hidden = false;
  } catch (error) {
    text(status, error.message); status.className = 'status error';
  }
});
