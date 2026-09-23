/**
 * SYNAPSE // BM25 Inverted Lexical Search Engine
 * Ultra-fast in-memory full-text search with Robertson-Spärck Jones IDF saturation.
 */

class BM25Index {
  /**
   * @param {Object} options 
   * @param {number} [options.k1=1.2] Term frequency saturation parameter
   * @param {number} [options.b=0.75] Document length normalization parameter
   */
  constructor(options = {}) {
    this.k1 = options.k1 ?? 1.2;
    this.b = options.b ?? 0.75;
    
    // Inverted Index: Map<token, Map<docId, termFreq>>
    this.invertedIndex = new Map();
    // Doc lengths: Map<docId, lengthInTokens>
    this.docLengths = new Map();
    // Total token count across all docs
    this.totalTokens = 0;
    // Total documents
    this.docCount = 0;

    this.stopwords = new Set([
      'a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
      'how', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the', 'this', 'to',
      'was', 'what', 'when', 'where', 'which', 'who', 'will', 'with'
    ]);
  }

  /**
   * Tokenizes text into normalized lexical terms.
   * Preserves programming identifiers, env vars ($PORT), dashes, and alphanumeric terms.
   * @param {string} text 
   * @returns {string[]}
   */
  tokenize(text) {
    if (!text || typeof text !== 'string') return [];
    const rawTokens = text.toLowerCase().match(/[\p{L}\p{N}_$-]+/gu) || [];
    return rawTokens.filter(t => !this.stopwords.has(t) && t.length > 1);
  }

  /**
   * Adds or updates a document in the lexical index.
   * @param {string|number} docId 
   * @param {string} text 
   */
  addDocument(docId, text) {
    if (this.docLengths.has(docId)) {
      this.removeDocument(docId);
    }

    const tokens = this.tokenize(text);
    const docLen = tokens.length;
    this.docLengths.set(docId, docLen);
    this.totalTokens += docLen;
    this.docCount++;

    const freqMap = new Map();
    for (const t of tokens) {
      freqMap.set(t, (freqMap.get(t) || 0) + 1);
    }

    for (const [token, freq] of freqMap.entries()) {
      if (!this.invertedIndex.has(token)) {
        this.invertedIndex.set(token, new Map());
      }
      this.invertedIndex.get(token).set(docId, freq);
    }
  }

  /**
   * Removes a document from the index.
   * @param {string|number} docId 
   */
  removeDocument(docId) {
    const docLen = this.docLengths.get(docId);
    if (docLen === undefined) return;

    this.totalTokens -= docLen;
    this.docLengths.delete(docId);
    this.docCount--;

    for (const [, posting] of this.invertedIndex.entries()) {
      if (posting.has(docId)) {
        posting.delete(docId);
      }
    }
  }

  /**
   * Average document length in tokens.
   * @returns {number}
   */
  get avgDocLength() {
    return this.docCount > 0 ? this.totalTokens / this.docCount : 1;
  }

  /**
   * Searches documents using BM25 scoring algorithm.
   * @param {string} query 
   * @param {number} [topK=10] 
   * @returns {Array<{ docId: string|number, score: number }>}
   */
  search(query, topK = 10) {
    const queryTokens = this.tokenize(query);
    if (queryTokens.length === 0 || this.docCount === 0) return [];

    const scores = new Map();
    const avgdl = this.avgDocLength;
    const N = this.docCount;

    for (const token of queryTokens) {
      const posting = this.invertedIndex.get(token);
      if (!posting) continue;

      const n = posting.size; // number of docs containing token
      // Robertson-Spärck Jones IDF
      const idf = Math.log(1 + (N - n + 0.5) / (n + 0.5));

      for (const [docId, tf] of posting.entries()) {
        const docLen = this.docLengths.get(docId) || avgdl;
        // BM25 term frequency saturation
        const numerator = tf * (this.k1 + 1);
        const denominator = tf + this.k1 * (1 - this.b + this.b * (docLen / avgdl));
        const termScore = idf * (numerator / denominator);

        scores.set(docId, (scores.get(docId) || 0) + termScore);
      }
    }

    const results = [];
    for (const [docId, score] of scores.entries()) {
      results.push({ docId, score });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { BM25Index };
}
