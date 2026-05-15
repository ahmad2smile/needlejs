/**
 * Needle Chrome Extension — Service Worker (Manifest V3)
 *
 * Handles model loading (lazily, on first request) and exposes
 * a chrome.runtime.onMessage API for content scripts and popup.
 *
 * Message protocol:
 *   Request:  { type: 'NEEDLE_GENERATE', query: string, tools: string|object[] }
 *   Response: { success: true, result: string } | { success: false, error: string }
 *
 *   Request:  { type: 'NEEDLE_STATUS' }
 *   Response: { loaded: boolean, loading: boolean }
 */

// CRITICAL: Must be set before any ort import/usage
// ort-wasm files are served from the extension's web_accessible_resources
self.ortWasmDir = null; // set in ensureLoaded() once chrome is available

// Dynamic import so bundlers handle onnxruntime-web correctly
import { configureOrtWasm, NeedleModel } from './needle.bundle.js';
import { NeedleTokenizer } from './needle.bundle.js';
import { generate } from './needle.bundle.js';

/** @type {NeedleModel|null} */
let model = null;
/** @type {NeedleTokenizer|null} */
let tokenizer = null;
let loading = false;
let loadError = null;

const MODEL_CACHE_NAME = 'needle-models-v1';

async function fetchWithCache(filename, url) {
  const cache = await caches.open(MODEL_CACHE_NAME);
  const cached = await cache.match(filename);
  if (cached) return cached.arrayBuffer();

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  await cache.put(filename, resp.clone());
  return resp.arrayBuffer();
}

async function ensureLoaded() {
  if (model && tokenizer) return;
  if (loadError) throw loadError;
  if (loading) {
    // Wait for the in-progress load
    await new Promise((resolve, reject) => {
      const interval = setInterval(() => {
        if (loadError) { clearInterval(interval); reject(loadError); }
        else if (model && tokenizer) { clearInterval(interval); resolve(); }
      }, 200);
    });
    return;
  }

  loading = true;
  try {
    const wasmDir = chrome.runtime.getURL('ort-wasm/');
    configureOrtWasm(wasmDir);

    // Model files: try bundled first (extension package), then HuggingFace
    const baseUrl = chrome.runtime.getURL('models/');
    const hfBase = 'https://huggingface.co/Cactus-Compute/needle/resolve/main';

    const [encBuf, decBuf, tokBuf] = await Promise.all([
      fetchWithCache('needle_encoder_fp16.onnx',
        baseUrl + 'needle_encoder_fp16.onnx').catch(
        () => fetchWithCache('needle_encoder_fp16.onnx',
          hfBase + '/needle_encoder_fp16.onnx')),
      fetchWithCache('needle_decoder_fp16.onnx',
        baseUrl + 'needle_decoder_fp16.onnx').catch(
        () => fetchWithCache('needle_decoder_fp16.onnx',
          hfBase + '/needle_decoder_fp16.onnx')),
      fetchWithCache('needle.model',
        baseUrl + 'tokenizer/needle.model').catch(
        () => fetchWithCache('needle.model',
          hfBase + '/tokenizer/needle.model')),
    ]);

    model = new NeedleModel();
    await model.loadFromBuffers(encBuf, decBuf, { useWebGPU: false });
    tokenizer = await NeedleTokenizer.fromBuffer(tokBuf);
  } catch (err) {
    loadError = err;
    loading = false;
    throw err;
  }
  loading = false;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === 'NEEDLE_STATUS') {
    sendResponse({ loaded: !!(model && tokenizer), loading });
    return false;
  }

  if (message.type === 'NEEDLE_GENERATE') {
    (async () => {
      try {
        await ensureLoaded();
        const result = await generate(model, tokenizer, message.query, message.tools, {
          maxGenLen: message.maxGenLen ?? 256,
          maxEncLen: message.maxEncLen ?? 1024,
          constrained: message.constrained ?? true,
        });
        sendResponse({ success: true, result });
      } catch (err) {
        sendResponse({ success: false, error: err.message });
      }
    })();
    return true; // Keep message channel open for async response
  }
});
