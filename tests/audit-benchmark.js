/**
 * SYNAPSE // Empirical Audit & Ground-Truth Benchmark
 * Measures:
 * 1. SQ8 Quantization Cosine Distortion across dimensions
 * 2. HNSW Recall@10 vs Exact Brute-Force Ground Truth
 * 3. Exact P50/P90/P99 latency distribution across scales (1K, 5K, 10K, 25K)
 * 4. Hybrid Search (RRF) vs pure Dense (HNSW) vs pure Sparse (BM25)
 */

const { HNSWIndex } = require('../engine/hnsw-index');
const { ScalarQuantizer } = require('../engine/quantizer');
const { VectorMath } = require('../engine/vector-math');
const { EpisodicMemoryStore } = require('../engine/memory-store');

console.log('\n======================================================');
console.log(' SYNAPSE // EMPIRICAL AUDIT & GROUND TRUTH BENCHMARK');
console.log('======================================================\n');

// ── 1. SQ8 Quantization Error Analysis ─────────────────────────
console.log('--- 1. SQ8 QUANTIZATION FIDELITY (vs Float32) ---');
const dims = [32, 64, 128, 384, 768];
for (const dim of dims) {
  let totalCosSim = 0;
  let maxCosDist = 0;
  let totalError = 0;
  const N = 2000;
  for (let i = 0; i < N; i++) {
    const raw = new Float32Array(dim);
    for (let d = 0; d < dim; d++) raw[d] = (Math.random() - 0.5) * 2;
    const norm = VectorMath.l2Normalize(raw);
    const q = ScalarQuantizer.quantize(norm);
    const deq = ScalarQuantizer.dequantize(q);
    const sim = VectorMath.cosineSimilarity(norm, deq);
    const dist = 1 - sim;
    totalCosSim += sim;
    totalError += dist;
    if (dist > maxCosDist) maxCosDist = dist;
  }
  const meanSim = totalCosSim / N;
  const meanDistErrPct = (totalError / N) * 100;
  const maxDistErrPct = maxCosDist * 100;
  console.log(`  Dim ${String(dim).padEnd(4)}: Mean Sim: ${meanSim.toFixed(6)} | Mean Distortion: ${meanDistErrPct.toFixed(4)}% | Max Distortion: ${maxDistErrPct.toFixed(4)}%`);
}

// ── 2. HNSW Recall@10 vs Brute-Force ─────────────────────────
console.log('\n--- 2. HNSW RECALL@10 & LATENCY (Ground Truth: Brute-Force) ---');
function benchmarkHNSW(N, dim = 64, useQuant = true, efSearch = 32) {
  const hnsw = new HNSWIndex({ M: 16, efConstruction: 64, efSearch, useQuantization: useQuant });
  const rawVectors = [];
  
  const tBuildStart = performance.now();
  for (let i = 0; i < N; i++) {
    const vec = new Float32Array(dim);
    for (let d = 0; d < dim; d++) vec[d] = (Math.random() - 0.5) * 2;
    const normalized = VectorMath.l2Normalize(vec);
    rawVectors.push(normalized);
    hnsw.insert(i, normalized);
  }
  const buildTimeMs = performance.now() - tBuildStart;

  // 100 Test queries
  const queryCount = 100;
  const testQueries = [];
  for (let i = 0; i < queryCount; i++) {
    const q = new Float32Array(dim);
    for (let d = 0; d < dim; d++) q[d] = (Math.random() - 0.5) * 2;
    testQueries.push(VectorMath.l2Normalize(q));
  }

  // Exact Brute Force Ground Truth
  const groundTruth = testQueries.map(q => {
    const scored = rawVectors.map((v, id) => ({ id, sim: VectorMath.cosineSimilarity(q, v, true) }));
    scored.sort((a, b) => b.sim - a.sim);
    return new Set(scored.slice(0, 10).map(x => x.id));
  });

  // HNSW Search Benchmark
  let totalHits = 0;
  const latencies = [];
  testQueries.forEach((q, idx) => {
    const t0 = performance.now();
    const results = hnsw.search(q, 10, efSearch);
    latencies.push(performance.now() - t0);
    const gt = groundTruth[idx];
    for (const r of results) {
      if (gt.has(r.id)) totalHits++;
    }
  });

  latencies.sort((a, b) => a - b);
  const recall = (totalHits / (queryCount * 10)) * 100;
  const p50 = latencies[Math.floor(queryCount * 0.50)].toFixed(3);
  const p90 = latencies[Math.floor(queryCount * 0.90)].toFixed(3);
  const p99 = latencies[Math.floor(queryCount * 0.99)].toFixed(3);
  const buildQPS = Math.round(N / (buildTimeMs / 1000));

  console.log(`  N=${String(N).padEnd(6)} | Dim=${String(dim).padEnd(3)} | Recall@10: ${recall.toFixed(1).padStart(5)}% | P50: ${p50.padStart(6)}ms | P90: ${p90.padStart(6)}ms | P99: ${p99.padStart(6)}ms | Build: ${buildQPS.toLocaleString()} v/s | MaxL: ${hnsw.maxLayer}`);
}

benchmarkHNSW(500, 64, true, 32);
benchmarkHNSW(1000, 64, true, 32);
benchmarkHNSW(5000, 64, true, 32);
benchmarkHNSW(10000, 64, true, 32);
benchmarkHNSW(10000, 64, true, 64); // Higher efSearch
benchmarkHNSW(10000, 64, false, 32); // Unquantized comparison

// ── 3. Hybrid RRF Case Studies ──────────────────────────────────
console.log('\n--- 3. HYBRID RETRIEVAL CASE STUDIES (RRF: Dense vs Sparse vs Hybrid) ---');
const store = new EpisodicMemoryStore({ dimension: 64, useQuantization: true });

const cases = [
  { id: 'doc1', text: 'Error: ECONNREFUSED on port 5432 while connecting to primary Postgres replica', metadata: { type: 'error' } },
  { id: 'doc2', text: 'Database connection pool exhausted after spike in concurrent user requests', metadata: { type: 'incident' } },
  { id: 'doc3', text: 'Authentication token CVE-2026-4421 security vulnerability advisory issued for auth service', metadata: { type: 'security' } },
  { id: 'doc4', text: 'Memory limit exceeded (OOMKilled) in worker pod container namespace', metadata: { type: 'infra' } },
  { id: 'doc5', text: 'Out of memory crash in worker process due to unbounded cache growth', metadata: { type: 'infra' } }
];

cases.forEach(c => store.remember(c));

function testQuery(q, desc) {
  console.log(`\n  Query: "${q}" (${desc})`);
  
  // Alpha = 1.0 (Pure Dense / Vector)
  const denseOnly = store.recall(q, { topK: 1, alpha: 1.0, applyTemporalDecay: false });
  // Alpha = 0.0 (Pure Sparse / BM25)
  const sparseOnly = store.recall(q, { topK: 1, alpha: 0.0, applyTemporalDecay: false });
  // Alpha = 0.65 (Hybrid RRF)
  const hybrid = store.recall(q, { topK: 1, alpha: 0.65, applyTemporalDecay: false });

  console.log(`    ↳ Pure Vector (alpha=1.0): [${denseOnly[0]?.id}] "${denseOnly[0]?.text.slice(0, 60)}..." (Sim: ${denseOnly[0]?.vectorSim})`);
  console.log(`    ↳ Pure BM25   (alpha=0.0): [${sparseOnly[0]?.id}] "${sparseOnly[0]?.text.slice(0, 60)}..." (BM25: ${sparseOnly[0]?.bm25Score})`);
  console.log(`    ↳ Hybrid RRF  (alpha=0.65): [${hybrid[0]?.id}] "${hybrid[0]?.text.slice(0, 60)}..." (Confidence: ${hybrid[0]?.confidence})`);
}

testQuery('port 5432', 'Exact Technical Identifier');
testQuery('RAM exhaustion and container crash', 'Semantic Paraphrase with Synonyms');
testQuery('CVE-2026-4421 auth token advisory', 'Mixed Exact Token + Semantic Intent');

console.log('\n======================================================\n');
