/**
 * Needle tokenizer: SentencePiece BPE wrapper.
 *
 * Loads the needle.model file and provides encode/decode.
 * Falls back to a pure-JS BPE implementation if sentencepiece-js
 * is unavailable (e.g. service worker environment issues with app-root-path).
 */

export const PAD_ID = 0;
export const EOS_ID = 1;
export const BOS_ID = 2;
export const UNK_ID = 3;
export const TOOL_CALL_ID = 4;
export const TOOLS_ID = 5;

export const DEFAULT_MAX_ENC_LEN = 1024;
export const DEFAULT_MAX_GEN_LEN = 256;

/** Convert camelCase/PascalCase/dot.notation to snake_case (mirrors needle's to_snake_case). */
export function toSnakeCase(name) {
  let s = name.replace(/[^a-zA-Z0-9_]+/g, '_');
  s = s.replace(/([a-z0-9])([A-Z])/g, '$1_$2');
  s = s.replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2');
  s = s.replace(/_+/g, '_');
  return s.toLowerCase().replace(/^_|_$/g, '');
}

/** Normalize tool names in a tools JSON array to snake_case. Returns {normalizedJson, nameMap}. */
export function normalizeTools(toolsJson) {
  let tools;
  try {
    tools = JSON.parse(toolsJson);
  } catch {
    return { normalizedJson: toolsJson, nameMap: {} };
  }
  const nameMap = {};
  for (const t of tools) {
    if (t && typeof t.name === 'string') {
      const snake = toSnakeCase(t.name);
      nameMap[snake] = t.name;
      t.name = snake;
    }
  }
  return { normalizedJson: JSON.stringify(tools), nameMap };
}

/** Replace snake_case tool names in model output back to original names. */
export function restoreToolNames(text, nameMap) {
  if (!nameMap || Object.keys(nameMap).length === 0) return text;
  try {
    const obj = JSON.parse(text);
    const fix = (o) => {
      if (Array.isArray(o)) return o.map(fix);
      if (o && typeof o === 'object' && typeof o.name === 'string') {
        o.name = nameMap[o.name] ?? o.name;
      }
      return o;
    };
    return JSON.stringify(fix(obj));
  } catch {
    // Fallback: string replacement, longest first
    for (const [snake, orig] of Object.entries(nameMap).sort((a, b) => b[0].length - a[0].length)) {
      text = text.replaceAll(snake, orig);
    }
    return text;
  }
}

/**
 * NeedleTokenizer wraps a SentencePiece model loaded from an ArrayBuffer.
 *
 * Usage:
 *   const tokenizer = await NeedleTokenizer.fromBuffer(modelBuffer);
 *   const ids = tokenizer.encode("hello world");
 *   const text = tokenizer.decode(ids);
 */
export class NeedleTokenizer {
  constructor(sp) {
    this._sp = sp;
  }

  /** Load tokenizer from an ArrayBuffer (the .model file bytes). */
  static async fromBuffer(buffer) {
    // Dynamic import so bundlers can tree-shake if not used
    const { SentencePieceProcessor } = await import('sentencepiece-js');
    const sp = new SentencePieceProcessor();
    await sp.loadFromBinaryArray(new Uint8Array(buffer));
    return new NeedleTokenizer(sp);
  }

  /** Load tokenizer from a URL (fetches the .model file). */
  static async fromUrl(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to fetch tokenizer: ${url} (${resp.status})`);
    const buf = await resp.arrayBuffer();
    return NeedleTokenizer.fromBuffer(buf);
  }

  get vocabSize() { return this._sp.vocabSize(); }

  encode(text) {
    return this._sp.encodeIds(text);
  }

  decode(ids) {
    return this._sp.decode(ids);
  }

  /**
   * Build the encoder input sequence:
   *   [query_tokens..., TOOLS_ID, tool_tokens...] truncated to maxEncLen
   */
  buildEncoderInput(query, toolsJson, maxEncLen = DEFAULT_MAX_ENC_LEN) {
    const queryIds = this.encode(query);
    const toolIds = this.encode(toolsJson);

    const maxQuery = maxEncLen - 2;
    const q = queryIds.slice(0, maxQuery);
    const remaining = maxEncLen - q.length - 1;
    const t = toolIds.slice(0, remaining);
    return [...q, TOOLS_ID, ...t];
  }
}
