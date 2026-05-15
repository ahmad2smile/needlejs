const queryEl = document.getElementById('query');
const toolsEl = document.getElementById('tools');
const runBtn = document.getElementById('run');
const statusEl = document.getElementById('status');
const outputEl = document.getElementById('output');
const errorEl = document.getElementById('error');

function setStatus(msg) { statusEl.textContent = msg; }
function showOutput(text) {
  outputEl.textContent = text;
  outputEl.style.display = 'block';
  errorEl.style.display = 'none';
}
function showError(msg) {
  errorEl.textContent = msg;
  errorEl.style.display = 'block';
  outputEl.style.display = 'none';
}

// Check model status on open
chrome.runtime.sendMessage({ type: 'NEEDLE_STATUS' }, (resp) => {
  if (resp?.loaded) setStatus('Model ready');
  else if (resp?.loading) setStatus('Model loading…');
  else setStatus('Model not loaded — will load on first run');
});

runBtn.addEventListener('click', async () => {
  const query = queryEl.value.trim();
  const toolsRaw = toolsEl.value.trim();
  if (!query) { showError('Enter a query.'); return; }
  if (!toolsRaw) { showError('Enter at least one tool definition.'); return; }

  // Validate tools JSON
  let tools;
  try {
    tools = JSON.parse(toolsRaw);
    if (!Array.isArray(tools)) throw new Error('Tools must be a JSON array');
  } catch (err) {
    showError(`Invalid tools JSON: ${err.message}`); return;
  }

  runBtn.disabled = true;
  setStatus('Running…');
  outputEl.style.display = 'none';
  errorEl.style.display = 'none';

  chrome.runtime.sendMessage({
    type: 'NEEDLE_GENERATE',
    query,
    tools: JSON.stringify(tools),
  }, (resp) => {
    runBtn.disabled = false;
    if (!resp) {
      showError('No response from background worker.');
      setStatus('');
      return;
    }
    if (resp.success) {
      showOutput(resp.result);
      setStatus('Done');
    } else {
      showError(resp.error || 'Unknown error');
      setStatus('Error');
    }
  });
});
