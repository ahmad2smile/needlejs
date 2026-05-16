/**
 * ONNX encoder + decoder sessions for Needle.
 *
 * Uses onnxruntime-web (WASM backend) which works in both Node and the browser.
 */

import * as ort from 'onnxruntime-web';

/**
 * Configure onnxruntime-web WASM paths. Only needed in the browser when the
 * WASM files aren't served from the same origin as the loader; in Node it
 * isn't required.
 *
 * @param {string} wasmDir - URL/path to directory containing ort-wasm*.wasm files
 */
export function configureOrtWasm(wasmDir) {
  if (!wasmDir.endsWith('/')) wasmDir += '/';
  ort.env.wasm.wasmPaths = wasmDir;
}

async function createSession(modelData, { useWebGPU = false } = {}) {
  const providers = useWebGPU ? ['webgpu', 'wasm'] : ['wasm'];
  return ort.InferenceSession.create(modelData, {
    executionProviders: providers,
    graphOptimizationLevel: 'all',
  });
}

export class NeedleModel {
  constructor() {
    this._encoder = null;
    this._decoder = null;
  }

  /**
   * Load encoder and decoder from ArrayBuffers / Uint8Arrays.
   */
  async loadFromBuffers(encoderData, decoderData, opts = {}) {
    [this._encoder, this._decoder] = await Promise.all([
      createSession(encoderData, opts),
      createSession(decoderData, opts),
    ]);
  }

  /**
   * Load encoder and decoder from filesystem paths (Node).
   */
  async loadFromPaths(encoderPath, decoderPath, opts = {}) {
    const { readFileSync } = await import('fs');
    return this.loadFromBuffers(readFileSync(encoderPath), readFileSync(decoderPath), opts);
  }

  /**
   * Load encoder and decoder from URLs (browser).
   */
  async loadFromUrls(encoderUrl, decoderUrl, opts = {}) {
    const [encResp, decResp] = await Promise.all([fetch(encoderUrl), fetch(decoderUrl)]);
    if (!encResp.ok) throw new Error(`Failed to fetch encoder: ${encoderUrl} (${encResp.status})`);
    if (!decResp.ok) throw new Error(`Failed to fetch decoder: ${decoderUrl} (${decResp.status})`);
    const [encBuf, decBuf] = await Promise.all([encResp.arrayBuffer(), decResp.arrayBuffer()]);
    return this.loadFromBuffers(encBuf, decBuf, opts);
  }

  /**
   * Run the encoder.
   * @param {Int32Array} inputIds - [enc_len] token IDs (batch=1 implicit)
   * @returns {Promise<ort.Tensor>} encoder_hidden_states [1, enc_len, d_model]
   */
  async encode(inputIds) {
    if (!this._encoder) throw new Error('Model not loaded');
    const tensor = new ort.Tensor('int32', inputIds, [1, inputIds.length]);
    const out = await this._encoder.run({ encoder_input_ids: tensor });
    return out.encoder_hidden_states;
  }

  /**
   * Run the decoder for all positions.
   * @param {Int32Array} decoderInputIds - [dec_len] token IDs
   * @param {ort.Tensor} encoderHiddenStates - [1, enc_len, d_model]
   * @returns {Promise<ort.Tensor>} logits [1, dec_len, vocab_size]
   */
  async decode(decoderInputIds, encoderHiddenStates) {
    if (!this._decoder) throw new Error('Model not loaded');
    const decTensor = new ort.Tensor('int32', decoderInputIds, [1, decoderInputIds.length]);
    const out = await this._decoder.run({
      decoder_input_ids: decTensor,
      encoder_hidden_states: encoderHiddenStates,
    });
    return out.logits;
  }

  get isLoaded() {
    return this._encoder !== null && this._decoder !== null;
  }
}
