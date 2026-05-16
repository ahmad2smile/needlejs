/**
 * Needle tokenizer: SentencePiece BPE wrapper.
 *
 * Wraps sentencepiece-js for encode/decode and parses the companion .vocab
 * file so we can expose vocabSize / idToPiece (the WASM build of
 * sentencepiece-js does not surface those methods).
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
    for (const [snake, orig] of Object.entries(nameMap).sort((a, b) => b[0].length - a[0].length)) {
      text = text.replaceAll(snake, orig);
    }
    return text;
  }
}

/** Parse a SentencePiece .vocab file (tab-separated piece<TAB>score) into pieces[]. */
function parseVocab(text) {
  return text.split('\n').filter((line) => line.length > 0).map((line) => {
    const tab = line.indexOf('\t');
    return tab === -1 ? line : line.slice(0, tab);
  });
}

/**
 * NeedleTokenizer wraps a SentencePiece model file and its companion vocab.
 *
 * Usage (Node):
 *   const tokenizer = await NeedleTokenizer.fromPath('/path/to/needle.model');
 *   const ids = tokenizer.encode("hello world");
 *   const text = tokenizer.decode(ids);
 */
export class NeedleTokenizer {
  constructor(sp, pieces) {
    this._sp = sp;
    this._pieces = pieces;
  }

  /**
   * Load tokenizer from a file path. Looks for the companion .vocab in the same
   * directory if vocabPath isn't given (e.g. needle.model → needle.vocab).
   */
  static async fromPath(modelPath, vocabPath = null) {
    const { SentencePieceProcessor } = await import('sentencepiece-js');
    const sp = new SentencePieceProcessor();
    await sp.load(modelPath);

    if (!vocabPath) vocabPath = modelPath.replace(/\.model$/, '.vocab');
    const { readFileSync } = await import('fs');
    const pieces = parseVocab(readFileSync(vocabPath, 'utf-8'));
    return new NeedleTokenizer(sp, pieces);
  }

  get vocabSize() { return this._pieces.length; }

  /** Return the piece string for vocab id i (e.g. '▁hello', '<0x41>', '<pad>'). */
  idToPiece(i) { return this._pieces[i] ?? ''; }

  encode(text) {
    return this._sp.encodeIds(text);
  }

  decode(ids) {
    return this._sp.decodeIds(ids);
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
