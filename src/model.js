/**
 * ONNX model sessions for Needle encoder and decoder.
 * Uses onnxruntime-web (WASM backend).
 */

import * as ort from 'onnxruntime-web';

/**
 * Configure onnxruntime-web WASM paths.
 * Must be called before creating any InferenceSession.
 *
 * @param {string} wasmDir - URL to directory containing ort-wasm*.wasm files
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
   * Load encoder and decoder from ArrayBuffers.
   * @param {ArrayBuffer} encoderData
   * @param {ArrayBuffer} decoderData
   * @param {{ useWebGPU?: boolean }} opts
   */
  async loadFromBuffers(encoderData, decoderData, opts = {}) {
    [this._encoder, this._decoder] = await Promise.all([
      createSession(encoderData, opts),
      createSession(decoderData, opts),
    ]);
  }

  /**
   * Load encoder and decoder from URLs.
   * @param {string} encoderUrl
   * @param {string} decoderUrl
   * @param {{ useWebGPU?: boolean }} opts
   */
  async loadFromUrls(encoderUrl, decoderUrl, opts = {}) {
    const [encResp, decResp] = await Promise.all([
      fetch(encoderUrl),
      fetch(decoderUrl),
    ]);
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

  /** Returns true if both sessions are loaded. */
  get isLoaded() {
    return this._encoder !== null && this._decoder !== null;
  }
}
