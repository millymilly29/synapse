/**
 * SYNAPSE // Hybrid Search & Reciprocal Rank Fusion (RRF)
 * Blends dense semantic vector retrieval (HNSW) with sparse exact keyword matching (BM25).
 */

class HybridRetriever {
  /**
   * @param {Object} options 
   * @param {number} [options.rrfK=60] RRF constant parameter to mitigate extreme rank bias
   * @param {number} [options.defaultAlpha=0.65] Dense vector weight vs sparse keyword weight (0 = pure BM25, 1 = pure HNSW)
   */
  constructor(options = {}) {
    this.rrfK = options.rrfK ?? 60;
    this.defaultAlpha = options.defaultAlpha ?? 0.65;
  }

  /**
   * Fuses Dense vector search results and Sparse BM25 lexical results via RRF.
   * 
   * @param {Array<{ id: string|number, similarity: number }>} denseResults 
   * @param {Array<{ docId: string|number, score: number }>} sparseResults 
   * @param {Object} [options]
   * @param {number} [options.alpha] Blend ratio (0.0 to 1.0)
   * @param {number} [options.topK=10]
   * @returns {Array<{ id: string|number, score: number, vectorRank: number|null, vectorSim: number|null, bm25Rank: number|null, bm25Score: number|null }>}
   */
  fuse(denseResults = [], sparseResults = [], options = {}) {
    const alpha = Math.max(0, Math.min(1.0, options.alpha ?? this.defaultAlpha));
    const topK = options.topK ?? 10;
    const k = this.rrfK;

    const merged = new Map();

    // 1. Process Dense Vector Ranks
    denseResults.forEach((item, index) => {
      const id = item.id;
      const rank = index + 1;
      const rrfContribution = alpha * (1 / (k + rank));

      merged.set(id, {
        id,
        score: rrfContribution,
        vectorRank: rank,
        vectorSim: item.similarity,
        bm25Rank: null,
        bm25Score: null
      });
    });

    // 2. Process Sparse BM25 Ranks
    sparseResults.forEach((item, index) => {
      const id = item.docId;
      const rank = index + 1;
      const rrfContribution = (1 - alpha) * (1 / (k + rank));

      if (merged.has(id)) {
        const existing = merged.get(id);
        existing.score += rrfContribution;
        existing.bm25Rank = rank;
        existing.bm25Score = item.score;
      } else {
        merged.set(id, {
          id,
          score: rrfContribution,
          vectorRank: null,
          vectorSim: null,
          bm25Rank: rank,
          bm25Score: item.score
        });
      }
    });

    // 3. Sort by aggregated RRF score
    const resultList = Array.from(merged.values());
    resultList.sort((a, b) => b.score - a.score);

    // Normalize final scores between 0.0 and 1.0 relative to max theoretical score
    const maxPossible = (alpha * (1 / (k + 1))) + ((1 - alpha) * (1 / (k + 1)));
    for (const r of resultList) {
      r.confidence = maxPossible > 0 ? Math.min(1.0, r.score / maxPossible) : 0;
    }

    return resultList.slice(0, topK);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { HybridRetriever };
}
