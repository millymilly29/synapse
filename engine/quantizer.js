/**
 * SYNAPSE // Scalar Quantizer (SQ8)
 * Compresses 32-bit floating-point embeddings into 8-bit quantized integer arrays (Uint8Array).
 * Delivers 4x memory compression (75% reduction) with >99% cosine fidelity.
 */

class ScalarQuantizer {
  /**
   * Quantizes a Float32Array vector into SQ8 format.
   * @param {Float32Array|number[]} vector 
   * @returns {{ qVec: Uint8Array, min: number, range: number, dim: number }}
   */
  static quantize(vector) {
    const dim = vector.length;
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < dim; i++) {
      const v = vector[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const range = (max - min) || 1e-7;
    const invRange = 255 / range;
    const qVec = new Uint8Array(dim);

    for (let i = 0; i < dim; i++) {
      const scaled = (vector[i] - min) * invRange;
      qVec[i] = Math.max(0, Math.min(255, Math.round(scaled)));
    }

    return {
      qVec,
      min,
      range,
      dim
    };
  }

  /**
   * Dequantizes an SQ8 vector back into a Float32Array.
   * @param {{ qVec: Uint8Array, min: number, range: number, dim: number }} quantized 
   * @returns {Float32Array}
   */
  static dequantize(quantized) {
    const { qVec, min, range, dim } = quantized;
    const out = new Float32Array(dim);
    const scale = range / 255;

    for (let i = 0; i < dim; i++) {
      out[i] = min + qVec[i] * scale;
    }

    return out;
  }

  /**
   * Computes approximate cosine similarity directly between a float query and an SQ8 quantized vector.
   * Asymmetric distance computation: avoids allocating dequantized float buffers on hot search path.
   * 
   * @param {Float32Array} floatVec (Normalized L2)
   * @param {{ qVec: Uint8Array, min: number, range: number, dim: number }} qTarget 
   * @returns {number}
   */
  static fastCosineSimilarity(floatVec, qTarget) {
    const { qVec, min, range, dim } = qTarget;
    const scale = range / 255;
    let dot = 0;
    let normBsq = 0;

    for (let i = 0; i < dim; i++) {
      const valB = min + qVec[i] * scale;
      dot += floatVec[i] * valB;
      normBsq += valB * valB;
    }

    if (normBsq === 0) return 0;
    // Assuming floatVec is normalized (norm = 1.0)
    return dot / Math.sqrt(normBsq);
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ScalarQuantizer };
}
