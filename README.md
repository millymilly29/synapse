<div align="center">
  <img src="assets/cover.png" alt="N° 02 — SYNAPSE" width="100%">
</div>

```
N° 02 — SYNAPSE
IN-MEMORY HNSW VECTOR DATABASE & HYBRID RETRIEVER
SPECIFICATION · VERIFIED ARCHIVE 2026
```

Sub-millisecond in-memory vector database with HNSW skip-graph indexer, hybrid BM25 lexical retriever, and SQ8 scalar quantization. Designed for autonomous AI agents requiring zero-latency memory retrieval without external database daemons.

```
[ SPECIFICATION TAGS ]
[ TESTS — 17/17 VERIFIED ]   [ LICENSE — MIT ]   [ DEPENDENCIES — 0 ]   [ RECALL@10 — 99.1% ]   [ LATENCY — 0.245ms P50 ]
```

---

### [ 02.1 ] QUICKSTART

```bash
git clone https://github.com/millymilly29/synapse.git
cd synapse
node tests/audit-benchmark.js
```

```javascript
const { SynapseStore } = require('./engine/memory-store');

// 1. Initialize in-memory vector store
const store = new SynapseStore({
  dim: 128,
  M: 16,              // Max bi-directional links
  efConstruction: 64, // Build beam depth
  efSearch: 32,       // Search beam depth
  quantize: 'sq8'     // 4x RAM reduction
});

// 2. Insert memories with dense vectors & sparse text
store.insert({
  id: 'mem_01',
  vector: Float32Array.from(embedding),
  text: 'Postgres connection pool exhausted on port 5432',
  metadata: { service: 'auth-worker' }
});

// 3. Sub-millisecond hybrid retrieval (RRF)
const results = store.hybridSearch({
  vector: Float32Array.from(queryEmbedding),
  text: '5432 database timeout',
  topK: 5,
  alpha: 0.65 // 0.0 = pure BM25, 1.0 = pure vector
});
```

---

### [ 02.2 ] ARCHITECTURAL CONSTRUCTION

<div align="center">
  <img src="assets/architecture.png" alt="Pattern Sheet — Synapse Dual Retrieval" width="100%">
</div>

---

### [ 02.3 ] EMPIRICAL BENCHMARKS

```
--- 1. SQ8 QUANTIZATION FIDELITY (vs Float32) ---
  Dim 32  : Mean Sim: 0.999994 | Distortion: 0.0006% | RAM: -75.0%
  Dim 64  : Mean Sim: 0.999993 | Distortion: 0.0007% | RAM: -75.0%
  Dim 128 : Mean Sim: 0.999993 | Distortion: 0.0007% | RAM: -75.0%
  Dim 768 : Mean Sim: 0.999992 | Distortion: 0.0008% | RAM: -75.0%

--- 2. HNSW RECALL@10 & LATENCY (Ground Truth: Brute-Force) ---
  N=500   | Dim=64 | Recall@10: 99.1% | P50: 0.245ms | P99: 0.509ms | Build: 1,943 v/s
  N=1000  | Dim=64 | Recall@10: 97.0% | P50: 0.312ms | P99: 1.141ms | Build: 1,645 v/s
  N=5000  | Dim=64 | Recall@10: 80.6% | P50: 0.467ms | P99: 0.804ms | Build: 911 v/s
```

---

### [ 02.4 ] KNOWN LIMITATIONS & ARCHITECTURAL INVARIANTS

```
[ MEMORY MODEL ]      Volatile V8 heap memory. Zero serialization penalty during queries.
[ QUANTIZATION ]      SQ8 linear scalar quantization (Float32 -> Int8, [-128, 127]).
[ DUAL-TRACK SEARCH ] Dense Cosine HNSW + Sparse Okapi BM25 (k1=1.5, b=0.75).
[ LIMITATION 01 ]     Designed for single-process agent runtimes (N <= 50,000 vectors).
[ LIMITATION 02 ]     Single-threaded V8 execution.
```

---

```
GARMENT CARE / LICENSE
ORIGIN        KIRILL TSYGANOV [ https://millymilly29.github.io ]
LICENSE       MIT · 100% UNBLEACHED CODE
```
