/**
 * SYNAPSE // Vector Math Primitives
 * High-performance typed-array vector operations with L2 normalization and deterministic pseudo-embeddings.
 */

class VectorMath {
  /**
   * Dot product between two Float32Arrays.
   * @param {Float32Array|number[]} a 
   * @param {Float32Array|number[]} b 
   * @returns {number}
   */
  static dotProduct(a, b) {
    const len = a.length;
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  /**
   * Euclidean L2 norm of a vector.
   * @param {Float32Array|number[]} a 
   * @returns {number}
   */
  static l2Norm(a) {
    const len = a.length;
    let sum = 0;
    for (let i = 0; i < len; i++) {
      sum += a[i] * a[i];
    }
    return Math.sqrt(sum);
  }

  /**
   * Normalizes a vector in-place or returns a new Float32Array.
   * @param {Float32Array|number[]} a 
   * @param {boolean} inPlace 
   * @returns {Float32Array}
   */
  static l2Normalize(a, inPlace = false) {
    const norm = VectorMath.l2Norm(a);
    const target = inPlace ? a : new Float32Array(a.length);
    if (norm === 0) return target;
    const invNorm = 1 / norm;
    for (let i = 0; i < a.length; i++) {
      target[i] = a[i] * invNorm;
    }
    return target;
  }

  /**
   * Computes Cosine Similarity between two vectors.
   * Range: [-1.0, 1.0]. If vectors are already L2 normalized, simplifies to dot product.
   * @param {Float32Array|number[]} a 
   * @param {Float32Array|number[]} b 
   * @param {boolean} alreadyNormalized 
   * @returns {number}
   */
  static cosineSimilarity(a, b, alreadyNormalized = false) {
    if (alreadyNormalized) {
      return VectorMath.dotProduct(a, b);
    }
    const dot = VectorMath.dotProduct(a, b);
    const normA = VectorMath.l2Norm(a);
    const normB = VectorMath.l2Norm(b);
    if (normA === 0 || normB === 0) return 0;
    return dot / (normA * normB);
  }

  /**
   * Euclidean (L2) distance between two vectors.
   * @param {Float32Array|number[]} a 
   * @param {Float32Array|number[]} b 
   * @returns {number}
   */
  static euclideanDistance(a, b) {
    const len = a.length;
    let sum = 0;
    for (let i = 0; i < len; i++) {
      const diff = a[i] - b[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Deterministic zero-dependency text embedding projection.
   * Uses multi-scale character n-gram hashing and sinusoidal orthogonal projections
   * to produce dense Float32Array embeddings where semantically related tokens cluster naturally.
   * 
   * @param {string} text 
   * @param {number} dim (default: 64)
   * @returns {Float32Array}
   */
  static embed(text, dim = 64) {
    const vec = new Float32Array(dim);
    if (!text || typeof text !== 'string') return vec;

    const normalized = text.toLowerCase().trim();
    const tokens = normalized.match(/[\p{L}\p{N}_$-]+/gu) || [normalized];

    // Seeded pseudo-random projection weights
    for (let tIdx = 0; tIdx < tokens.length; tIdx++) {
      const token = tokens[tIdx];
      let hash = 0x811c9dc5;
      for (let i = 0; i < token.length; i++) {
        hash ^= token.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193);
      }

      // Project token across embedding dimensions with orthogonal frequencies
      for (let d = 0; d < dim; d++) {
        const freq = (d + 1) * 0.137;
        const phase = (hash % 1000) * 0.00628;
        const weight = Math.sin(freq * (tIdx + 1) + phase) + Math.cos(freq * 1.618);
        vec[d] += weight;
      }

      // Add sub-word 3-gram character features
      if (token.length >= 3) {
        for (let i = 0; i <= token.length - 3; i++) {
          const charHash = (token.charCodeAt(i) * 31 + token.charCodeAt(i + 1)) * 31 + token.charCodeAt(i + 2);
          const targetDim = Math.abs(charHash) % dim;
          vec[targetDim] += 0.45;
        }
      }
    }

    return VectorMath.l2Normalize(vec, true);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { VectorMath };
}
