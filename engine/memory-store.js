/**
 * SYNAPSE // Episodic Agent Memory Store
 * Full-lifecycle memory engine combining HNSW graph indexing, BM25 keyword matching,
 * temporal decay scoring, metadata filtering, and working memory ring buffer.
 */

const { VectorMath } = require('./vector-math');
const { HNSWIndex } = require('./hnsw-index');
const { BM25Index } = require('./bm25');
const { HybridRetriever } = require('./hybrid-search');

class EpisodicMemoryStore {
  /**
   * @param {Object} [options]
   * @param {number} [options.dimension=64] Vector dimension
   * @param {number} [options.workingMemoryCapacity=10] Size of working memory buffer
   * @param {number} [options.decayLambda=0.00005] Half-life decay constant for temporal aging
   * @param {boolean} [options.useQuantization=true] Enable SQ8 4x memory compression
   */
  constructor(options = {}) {
    this.dimension = options.dimension ?? 64;
    this.workingMemoryCapacity = options.workingMemoryCapacity ?? 10;
    this.decayLambda = options.decayLambda ?? 0.00005; // ~exponential decay
    this.useQuantization = options.useQuantization ?? true;

    this.hnsw = new HNSWIndex({
      M: 16,
      efConstruction: 64,
      efSearch: 32,
      useQuantization: this.useQuantization
    });

    this.bm25 = new BM25Index({ k1: 1.2, b: 0.75 });
    this.retriever = new HybridRetriever({ defaultAlpha: 0.65 });

    // Documents: Map<id, { id, text, metadata, vector, timestamp, accessCount, lastAccessed }>
    this.documents = new Map();
    // Working memory ring buffer (FIFO)
    this.workingMemory = [];
  }

  /**
   * Stores a new memory into the episodic index.
   * 
   * @param {Object} memory 
   * @param {string|number} memory.id Unique identifier
   * @param {string} memory.text Memory content
   * @param {Float32Array|number[]} [memory.vector] Optional precomputed vector; auto-embedded if missing
   * @param {Object} [memory.metadata] Arbitrary key-value metadata tags
   * @param {number} [memory.timestamp] Epoch timestamp (default: Date.now())
   * @returns {Object} Stored record
   */
  remember(memory) {
    if (!memory || !memory.id || !memory.text) {
      throw new Error('Memory must have both id and text properties.');
    }

    const id = memory.id;
    const text = String(memory.text);
    const timestamp = memory.timestamp ?? Date.now();
    const metadata = memory.metadata ?? {};

    // 1. Compute or normalize embedding vector
    const vector = memory.vector 
      ? VectorMath.l2Normalize(memory.vector instanceof Float32Array ? memory.vector : new Float32Array(memory.vector))
      : VectorMath.embed(text, this.dimension);

    // 2. Index in HNSW
    this.hnsw.insert(id, vector);

    // 3. Index in BM25
    this.bm25.addDocument(id, text);

    // 4. Save record
    const record = {
      id,
      text,
      metadata,
      vector,
      timestamp,
      accessCount: 0,
      lastAccessed: timestamp
    };
    this.documents.set(id, record);

    // 5. Push to Working Memory FIFO
    this.workingMemory.unshift(record);
    if (this.workingMemory.length > this.workingMemoryCapacity) {
      this.workingMemory.pop();
    }

    return record;
  }

  /**
   * Searches memory using hybrid dense + sparse retrieval with temporal decay and metadata filters.
   * 
   * @param {string} query 
   * @param {Object} [options]
   * @param {number} [options.topK=5]
   * @param {number} [options.alpha=0.65] Dense/sparse blend ratio
   * @param {Object} [options.filter] Key-value equality filter for metadata
   * @param {boolean} [options.applyTemporalDecay=true] Apply exponential time-decay penalty
   * @returns {Array<{ id: any, text: string, metadata: Object, confidence: number, score: number, vectorSim: number, bm25Score: number, decayMultiplier: number }>}
   */
  recall(query, options = {}) {
    const topK = options.topK ?? 5;
    const alpha = options.alpha ?? 0.65;
    const filter = options.filter ?? null;
    const applyDecay = options.applyTemporalDecay ?? true;

    const queryVec = VectorMath.embed(query, this.dimension);
    const now = Date.now();

    // 1. Query HNSW (dense)
    const denseCandidates = this.hnsw.search(queryVec, topK * 4);

    // 2. Query BM25 (sparse)
    const sparseCandidates = this.bm25.search(query, topK * 4);

    // 3. Hybrid Fusion
    const fused = this.retriever.fuse(denseCandidates, sparseCandidates, { alpha, topK: topK * 4 });

    // 4. Filter & Apply Temporal Decay
    const results = [];
    for (const item of fused) {
      const doc = this.documents.get(item.id);
      if (!doc) continue;

      // Metadata filtering
      if (filter) {
        let match = true;
        for (const [k, v] of Object.entries(filter)) {
          if (doc.metadata[k] !== v) {
            match = false;
            break;
          }
        }
        if (!match) continue;
      }

      // Temporal decay calculation: e^(-lambda * dt_seconds)
      let decay = 1.0;
      if (applyDecay && this.decayLambda > 0) {
        const ageInSec = Math.max(0, (now - doc.timestamp) / 1000);
        decay = Math.exp(-this.decayLambda * ageInSec);
      }

      const finalScore = item.confidence * decay;

      doc.accessCount++;
      doc.lastAccessed = now;

      results.push({
        id: doc.id,
        text: doc.text,
        metadata: doc.metadata,
        confidence: Math.round(finalScore * 1000) / 1000,
        rawScore: item.score,
        vectorSim: item.vectorSim ? Math.round(item.vectorSim * 1000) / 1000 : 0,
        bm25Score: item.bm25Score ? Math.round(item.bm25Score * 1000) / 1000 : 0,
        decayMultiplier: Math.round(decay * 1000) / 1000,
        timestamp: doc.timestamp
      });
    }

    results.sort((a, b) => b.confidence - a.confidence);
    return results.slice(0, topK);
  }

  /**
   * Returns recent working memory buffer items.
   * @returns {Array<Object>}
   */
  getWorkingMemory() {
    return this.workingMemory.map(m => ({
      id: m.id,
      text: m.text,
      metadata: m.metadata,
      timestamp: m.timestamp
    }));
  }

  /**
   * Diagnostic statistics about the store.
   * @returns {Object}
   */
  getTelemetry() {
    const totalDocs = this.documents.size;
    const rawVectorBytes = totalDocs * this.dimension * 4; // 4 bytes per float32
    const quantizedBytes = totalDocs * (this.dimension + 8); // 1 byte per dim + min/range floats
    const bytesSaved = this.useQuantization ? (rawVectorBytes - quantizedBytes) : 0;
    const compressionRatio = this.useQuantization ? (rawVectorBytes / Math.max(1, quantizedBytes)).toFixed(2) : '1.00';

    return {
      totalMemories: totalDocs,
      dimension: this.dimension,
      hnswMaxLayer: this.hnsw.maxLayer,
      quantizationEnabled: this.useQuantization,
      rawVectorBytes,
      activeStorageBytes: this.useQuantization ? quantizedBytes : rawVectorBytes,
      bytesSaved,
      compressionRatio: `${compressionRatio}x`,
      workingMemorySize: this.workingMemory.length
    };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { EpisodicMemoryStore };
}
