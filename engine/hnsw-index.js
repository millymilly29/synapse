/**
 * SYNAPSE // Hierarchical Navigable Small World (HNSW) Index
 * Logarithmic O(log N) approximate nearest neighbor vector search with multi-layer skip graphs.
 */

const { VectorMath } = require('./vector-math');
const { ScalarQuantizer } = require('./quantizer');

class HNSWIndex {
  /**
   * @param {Object} options 
   * @param {number} [options.M=16] Max number of bidirectional links per node per layer (2*M on layer 0)
   * @param {number} [options.efConstruction=64] Size of dynamic candidate list during construction
   * @param {number} [options.efSearch=32] Size of dynamic candidate list during search
   * @param {boolean} [options.useQuantization=true] Enable SQ8 vector compression
   */
  constructor(options = {}) {
    this.M = options.M ?? 16;
    this.M0 = 2 * this.M;
    this.efConstruction = options.efConstruction ?? 64;
    this.efSearch = options.efSearch ?? 32;
    this.mL = 1 / Math.log(this.M);
    this.useQuantization = options.useQuantization ?? true;

    // Node storage: Map<id, { id, vector: Float32Array, qVector: Object, layer: number, friends: Array<Set<number|string>> }>
    this.nodes = new Map();
    this.enterPointId = null;
    this.maxLayer = -1;
    this.size = 0;
  }

  /**
   * Generates a random layer for a new node using exponential decay.
   * @returns {number}
   */
  _randomLayer() {
    const r = Math.random();
    if (r === 0) return 0;
    return Math.floor(-Math.log(r) * this.mL);
  }

  /**
   * Computes distance between query vector and a node (0.0 = identical, 2.0 = opposite).
   * @param {Float32Array} queryVec (L2 normalized)
   * @param {Object} node 
   * @returns {number}
   */
  _distance(queryVec, node) {
    if (this.useQuantization && node.qVector) {
      const sim = ScalarQuantizer.fastCosineSimilarity(queryVec, node.qVector);
      return 1.0 - sim;
    }
    const sim = VectorMath.cosineSimilarity(queryVec, node.vector, true);
    return 1.0 - sim;
  }

  /**
   * Distance between two existing nodes.
   * @param {Object} nodeA 
   * @param {Object} nodeB 
   * @returns {number}
   */
  _nodeDistance(nodeA, nodeB) {
    return 1.0 - VectorMath.cosineSimilarity(nodeA.vector, nodeB.vector, true);
  }

  /**
   * Inserts a vector into the HNSW graph.
   * @param {string|number} id 
   * @param {Float32Array|number[]} rawVector 
   */
  insert(id, rawVector) {
    const vector = VectorMath.l2Normalize(
      rawVector instanceof Float32Array ? rawVector : new Float32Array(rawVector)
    );

    const targetLayer = this._randomLayer();
    const node = {
      id,
      vector,
      qVector: this.useQuantization ? ScalarQuantizer.quantize(vector) : null,
      layer: targetLayer,
      friends: [] // friends[l] is Set of neighbor IDs at layer l
    };

    for (let l = 0; l <= targetLayer; l++) {
      node.friends[l] = new Set();
    }

    if (this.enterPointId === null) {
      this.enterPointId = id;
      this.maxLayer = targetLayer;
      this.nodes.set(id, node);
      this.size++;
      return;
    }

    let currObj = this.nodes.get(this.enterPointId);
    let currDist = this._distance(vector, currObj);

    // 1. Greedy search from top layer down to targetLayer + 1
    for (let l = this.maxLayer; l > targetLayer; l--) {
      let changed = true;
      while (changed) {
        changed = false;
        const neighbors = currObj.friends[l] || [];
        for (const neighborId of neighbors) {
          const neighbor = this.nodes.get(neighborId);
          if (!neighbor) continue;
          const d = this._distance(vector, neighbor);
          if (d < currDist) {
            currDist = d;
            currObj = neighbor;
            changed = true;
          }
        }
      }
    }

    // 2. Multi-layer beam search and neighbor attachment from min(maxLayer, targetLayer) down to 0
    let enterCandidates = [{ id: currObj.id, dist: currDist }];

    for (let l = Math.min(this.maxLayer, targetLayer); l >= 0; l--) {
      const candidates = this._searchLayer(vector, enterCandidates, this.efConstruction, l);

      // Select M closest neighbors (or M0 for layer 0)
      const maxM = (l === 0) ? this.M0 : this.M;
      const neighbors = candidates.slice(0, maxM);

      // Connect node to neighbors bidirectionally
      for (const { id: neighborId } of neighbors) {
        node.friends[l].add(neighborId);
        const neighborNode = this.nodes.get(neighborId);
        if (neighborNode) {
          neighborNode.friends[l].add(id);

          // Prune neighbor's links if exceeding limit
          if (neighborNode.friends[l].size > maxM) {
            this._pruneNeighbors(neighborNode, l, maxM);
          }
        }
      }

      enterCandidates = candidates;
    }

    if (targetLayer > this.maxLayer) {
      this.maxLayer = targetLayer;
      this.enterPointId = id;
    }

    this.nodes.set(id, node);
    this.size++;
  }

  /**
   * Prunes neighbors of a node to keep only the closest maxM nodes.
   * @param {Object} node 
   * @param {number} layer 
   * @param {number} maxM 
   */
  _pruneNeighbors(node, layer, maxM) {
    const list = Array.from(node.friends[layer]).map(nId => {
      const neighbor = this.nodes.get(nId);
      return { id: nId, dist: neighbor ? this._nodeDistance(node, neighbor) : Infinity };
    });

    list.sort((a, b) => a.dist - b.dist);
    node.friends[layer] = new Set(list.slice(0, maxM).map(x => x.id));
  }

  /**
   * Beam search on a single layer.
   * @param {Float32Array} queryVec 
   * @param {Array<{ id: any, dist: number }>} enterPoints 
   * @param {number} ef 
   * @param {number} layer 
   * @returns {Array<{ id: any, dist: number }>}
   */
  _searchLayer(queryVec, enterPoints, ef, layer) {
    const visited = new Set();
    const candidates = []; // Min-heap conceptually, kept sorted by dist asc
    const found = [];      // Best candidates, sorted by dist asc

    for (const ep of enterPoints) {
      visited.add(ep.id);
      candidates.push(ep);
      found.push(ep);
    }
    candidates.sort((a, b) => a.dist - b.dist);
    found.sort((a, b) => a.dist - b.dist);

    while (candidates.length > 0) {
      const curr = candidates.shift();
      const furthestFoundDist = found[found.length - 1].dist;

      if (curr.dist > furthestFoundDist && found.length >= ef) {
        break;
      }

      const currNode = this.nodes.get(curr.id);
      if (!currNode) continue;
      const neighbors = currNode.friends[layer] || [];

      for (const neighborId of neighbors) {
        if (visited.has(neighborId)) continue;
        visited.add(neighborId);

        const neighborNode = this.nodes.get(neighborId);
        if (!neighborNode) continue;
        const d = this._distance(queryVec, neighborNode);

        if (d < furthestFoundDist || found.length < ef) {
          const item = { id: neighborId, dist: d };
          candidates.push(item);
          candidates.sort((a, b) => a.dist - b.dist);

          found.push(item);
          found.sort((a, b) => a.dist - b.dist);
          if (found.length > ef) {
            found.pop();
          }
        }
      }
    }

    return found;
  }

  /**
   * Performs Approximate Nearest Neighbor (ANN) search.
   * Returns topK items sorted by highest cosine similarity (lowest distance).
   * @param {Float32Array|number[]} query 
   * @param {number} [topK=10] 
   * @param {number} [efSearch] 
   * @returns {Array<{ id: string|number, similarity: number, distance: number }>}
   */
  search(query, topK = 10, efSearch = this.efSearch) {
    if (this.size === 0 || this.enterPointId === null) return [];

    const queryVec = VectorMath.l2Normalize(
      query instanceof Float32Array ? query : new Float32Array(query)
    );

    let currObj = this.nodes.get(this.enterPointId);
    let currDist = this._distance(queryVec, currObj);

    // 1. Greedy search from top down to layer 1
    for (let l = this.maxLayer; l > 0; l--) {
      let changed = true;
      while (changed) {
        changed = false;
        const neighbors = currObj.friends[l] || [];
        for (const neighborId of neighbors) {
          const neighbor = this.nodes.get(neighborId);
          if (!neighbor) continue;
          const d = this._distance(queryVec, neighbor);
          if (d < currDist) {
            currDist = d;
            currObj = neighbor;
            changed = true;
          }
        }
      }
    }

    // 2. Beam search on bottom layer 0
    const candidates = this._searchLayer(queryVec, [{ id: currObj.id, dist: currDist }], Math.max(efSearch, topK), 0);

    return candidates.slice(0, topK).map(c => ({
      id: c.id,
      distance: c.dist,
      similarity: Math.max(0, Math.min(1.0, 1.0 - c.dist))
    }));
  }

  /**
   * Search with hop-by-hop diagnostic trail across layers (for visualizer).
   * @param {Float32Array|number[]} query 
   * @param {number} [topK=5] 
   * @returns {{ results: Array, trail: Array<{ layer: number, from: any, to: any, dist: number }> }}
   */
  searchWithTrail(query, topK = 5) {
    const trail = [];
    if (this.size === 0 || this.enterPointId === null) {
      return { results: [], trail };
    }

    const queryVec = VectorMath.l2Normalize(
      query instanceof Float32Array ? query : new Float32Array(query)
    );

    let currObj = this.nodes.get(this.enterPointId);
    let currDist = this._distance(queryVec, currObj);

    for (let l = this.maxLayer; l > 0; l--) {
      let changed = true;
      while (changed) {
        changed = false;
        const neighbors = currObj.friends[l] || [];
        for (const neighborId of neighbors) {
          const neighbor = this.nodes.get(neighborId);
          if (!neighbor) continue;
          const d = this._distance(queryVec, neighbor);
          if (d < currDist) {
            trail.push({ layer: l, from: currObj.id, to: neighbor.id, dist: d });
            currDist = d;
            currObj = neighbor;
            changed = true;
          }
        }
      }
    }

    const candidates = this._searchLayer(queryVec, [{ id: currObj.id, dist: currDist }], Math.max(this.efSearch, topK), 0);
    const results = candidates.slice(0, topK).map(c => ({
      id: c.id,
      distance: c.dist,
      similarity: Math.max(0, Math.min(1.0, 1.0 - c.dist))
    }));

    return { results, trail };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { HNSWIndex };
}
