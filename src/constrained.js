/**
 * Grammar-constrained decoding for Needle tool-call generation.
 *
 * Direct port of needle/model/constrained.py.
 *
 * Constrains the decoder to only produce valid tool names and argument keys
 * by tracking position in the output JSON and masking invalid tokens via a
 * character-level trie built from the tool definitions.
 *
 * Needle output format (compact JSON):
 *   {"name":"tool_name","arguments":{"key1":value1,"key2":value2}}
 */

// ---------------------------------------------------------------------------
// Trie
// ---------------------------------------------------------------------------

export class TrieNode {
  constructor() {
    /** @type {Map<string, TrieNode>} */
    this.children = new Map();
    this.isTerminal = false;
  }
}

export class Trie {
  constructor() {
    this.root = new TrieNode();
  }

  insert(word) {
    let node = this.root;
    for (const ch of word) {
      if (!node.children.has(ch)) node.children.set(ch, new TrieNode());
      node = node.children.get(ch);
    }
    node.isTerminal = true;
  }

  /** Return the trie node after consuming prefix, or null if off-trie. */
  getNode(prefix) {
    let node = this.root;
    for (const ch of prefix) {
      if (!node.children.has(ch)) return null;
      node = node.children.get(ch);
    }
    return node;
  }

  get words() {
    const result = [];
    const dfs = (node, path) => {
      if (node.isTerminal) result.push(path);
      for (const [ch, child] of node.children) dfs(child, path + ch);
    };
    dfs(this.root, '');
    return result;
  }
}

// ---------------------------------------------------------------------------
// ToolConstraints
// ---------------------------------------------------------------------------

export class ToolConstraints {
  /** @param {string} toolsJson - JSON array of tool definitions */
  constructor(toolsJson) {
    this.nameTrie = new Trie();
    /** @type {Map<string, Trie>} */
    this.paramTries = new Map();

    let tools;
    try {
      tools = JSON.parse(toolsJson);
    } catch {
      tools = [];
    }
    if (!Array.isArray(tools)) tools = [];

    for (const tool of tools) {
      if (!tool || typeof tool.name !== 'string' || !tool.name) continue;
      this.nameTrie.insert(tool.name);

      const params = tool.parameters;
      if (params && typeof params === 'object') {
        const paramTrie = new Trie();
        for (const [key, val] of Object.entries(params)) {
          if (typeof val === 'object') paramTrie.insert(key);
        }
        this.paramTries.set(tool.name, paramTrie);
      }
    }
  }

  getParamTrie(funcName) {
    return this.paramTries.get(funcName) ?? null;
  }
}

// ---------------------------------------------------------------------------
// JsonStateMachine
// ---------------------------------------------------------------------------

export const JsonState = /** @type {const} */ ({
  FREE: 'FREE',
  IN_NAME: 'IN_NAME',
  IN_ARG_KEY: 'IN_ARG_KEY',
});

export class JsonStateMachine {
  constructor() {
    this.state = JsonState.FREE;
    this.buffer = '';
    this.constrainedBuf = '';
    this.currentFunction = '';
    this.inArguments = false;
    this.argumentsDepth = 0;
    this.nestingDepth = 0;
    this.inString = false;
    this.prevCharEscape = false;
  }

  feed(text) {
    for (const ch of text) this._feedChar(ch);
  }

  _feedChar(ch) {
    if (this.state === JsonState.IN_NAME || this.state === JsonState.IN_ARG_KEY) {
      if (ch === '"') {
        if (this.state === JsonState.IN_NAME) this.currentFunction = this.constrainedBuf;
        this.constrainedBuf = '';
        this.state = JsonState.FREE;
      } else {
        this.constrainedBuf += ch;
      }
      this.buffer += ch;
      return;
    }

    this.buffer += ch;

    if (this.inString) {
      if (this.prevCharEscape) { this.prevCharEscape = false; return; }
      if (ch === '\\') { this.prevCharEscape = true; return; }
      if (ch === '"') this.inString = false;
      return;
    }

    if (ch === '{' || ch === '[') {
      this.nestingDepth++;
    } else if (ch === '}' || ch === ']') {
      this.nestingDepth = Math.max(0, this.nestingDepth - 1);
      if (ch === '}' && this.inArguments && this.nestingDepth < this.argumentsDepth) {
        this.inArguments = false;
      }
      return;
    }

    if (this.buffer.endsWith('"name":"') && !this.inArguments) {
      this.state = JsonState.IN_NAME;
      this.constrainedBuf = '';
      return;
    }

    if (this.buffer.endsWith('"arguments":{')) {
      this.inArguments = true;
      this.argumentsDepth = this.nestingDepth;
      return;
    }

    if (this.inArguments &&
        this.nestingDepth === this.argumentsDepth &&
        this._atArgKeyStart()) {
      this.state = JsonState.IN_ARG_KEY;
      this.constrainedBuf = '';
      return;
    }

    if (ch === '"' && this._isValueQuote()) {
      this.inString = true;
    }
  }

  _atArgKeyStart() {
    if (this.buffer.length < 2) return false;
    const tail = this.buffer.slice(-2);
    return tail === '{"' || tail === ',"';
  }

  _isValueQuote() {
    for (let j = this.buffer.length - 2; j >= 0; j--) {
      const c = this.buffer[j];
      if (c === ' ' || c === '\t' || c === '\n' || c === '\r') continue;
      return c === ':';
    }
    return false;
  }
}

// ---------------------------------------------------------------------------
// Token strings and index
// ---------------------------------------------------------------------------

/**
 * Map each vocab ID to the characters it contributes to decoded text.
 * SentencePiece uses ▁ (U+2581) as word-boundary marker → space.
 * @param {import('./tokenizer.js').NeedleTokenizer} tokenizer
 * @returns {string[]}
 */
export function buildTokenStrings(tokenizer) {
  const vocabSize = tokenizer.vocabSize;
  const strings = new Array(vocabSize).fill('');
  for (let i = 0; i < vocabSize; i++) {
    try {
      const piece = tokenizer.idToPiece(i);
      if (!piece) continue;
      // Replace ▁ with space; skip control/byte tokens
      if (piece.startsWith('<') && piece.endsWith('>')) {
        // Special or byte token — attempt to decode as single byte
        if (piece.startsWith('<0x')) {
          try {
            const byte = parseInt(piece.slice(3, -1), 16);
            strings[i] = String.fromCharCode(byte);
          } catch { /* ignore */ }
        }
        // else leave as empty string
      } else {
        strings[i] = piece.replace(/▁/g, ' ');
      }
    } catch { /* ignore */ }
  }
  return strings;
}

export class TokenIndex {
  /** @param {string[]} tokenStrings */
  constructor(tokenStrings) {
    /** @type {Map<string, number[]>} */
    this._index = new Map();
    for (let tid = 0; tid < tokenStrings.length; tid++) {
      const s = tokenStrings[tid];
      if (!s) continue;
      const first = s[0];
      if (!this._index.has(first)) this._index.set(first, []);
      this._index.get(first).push(tid);
    }
  }

  candidatesFor(firstChar) {
    return this._index.get(firstChar) ?? [];
  }
}

// ---------------------------------------------------------------------------
// Token validity check and logit masking
// ---------------------------------------------------------------------------

function checkTokenValid(tokenText, trieNode) {
  let node = trieNode;
  for (const ch of tokenText) {
    if (ch === '"') return node.isTerminal;
    if (!node.children.has(ch)) return false;
    node = node.children.get(ch);
  }
  return true;
}

/**
 * Mask logits so only valid tokens survive.
 * Sets invalid positions to -Infinity.
 * @param {Float32Array} logits
 * @param {TrieNode} trieNode
 * @param {string[]} tokenStrings
 * @param {TokenIndex} tokenIndex
 * @returns {Float32Array}
 */
export function applyConstraints(logits, trieNode, tokenStrings, tokenIndex) {
  const mask = new Uint8Array(logits.length); // 0 = blocked, 1 = allowed

  const validFirstChars = new Set(trieNode.children.keys());
  if (trieNode.isTerminal) validFirstChars.add('"');

  for (const firstChar of validFirstChars) {
    for (const tid of tokenIndex.candidatesFor(firstChar)) {
      if (!mask[tid] && checkTokenValid(tokenStrings[tid], trieNode)) {
        mask[tid] = 1;
      }
    }
  }

  let anyValid = false;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { anyValid = true; break; }
  if (!anyValid) return logits; // fallback: unconstrained

  const result = new Float32Array(logits);
  for (let i = 0; i < result.length; i++) {
    if (!mask[i]) result[i] = -Infinity;
  }
  return result;
}

// ---------------------------------------------------------------------------
// ConstrainedDecoder
// ---------------------------------------------------------------------------

export class ConstrainedDecoder {
  /**
   * @param {ToolConstraints[]} toolConstraintsList - one per batch item
   * @param {string[]} tokenStrings
   * @param {TokenIndex} tokenIndex
   */
  constructor(toolConstraintsList, tokenStrings, tokenIndex) {
    this.batchSize = toolConstraintsList.length;
    this.toolConstraints = toolConstraintsList;
    this.machines = toolConstraintsList.map(() => new JsonStateMachine());
    this.tokenStrings = tokenStrings;
    this.tokenIndex = tokenIndex;
  }

  isActive(batchIdx) {
    return this.machines[batchIdx].state !== JsonState.FREE;
  }

  constrainLogits(logits, batchIdx) {
    const machine = this.machines[batchIdx];
    const tc = this.toolConstraints[batchIdx];

    if (machine.state === JsonState.FREE) return logits;

    let trie;
    if (machine.state === JsonState.IN_NAME) {
      trie = tc.nameTrie;
    } else if (machine.state === JsonState.IN_ARG_KEY) {
      trie = tc.getParamTrie(machine.currentFunction);
      if (!trie) return logits;
    } else {
      return logits;
    }

    const node = trie.getNode(machine.constrainedBuf);
    if (!node) return logits; // off-trie fallback

    return applyConstraints(logits, node, this.tokenStrings, this.tokenIndex);
  }

  update(batchIdx, tokenId) {
    const text = this.tokenStrings[tokenId];
    if (text) this.machines[batchIdx].feed(text);
  }
}

/**
 * Build a ConstrainedDecoder for a batch of examples.
 * @param {string[]} toolsJsonList
 * @param {import('./tokenizer.js').NeedleTokenizer} tokenizer
 */
export function buildConstrainedDecoder(toolsJsonList, tokenizer) {
  const tokenStrings = buildTokenStrings(tokenizer);
  const tokenIndex = new TokenIndex(tokenStrings);
  const tcList = toolsJsonList.map(tj => new ToolConstraints(tj));
  return new ConstrainedDecoder(tcList, tokenStrings, tokenIndex);
}
