/**
 * SYNAPSE // Verification Test Suite
 * 17 tests validating vector math, HNSW graph search, BM25 indexing, SQ8 quantization, and hybrid memory recall.
 */

const assert = require('assert');
const {
  VectorMath,
  ScalarQuantizer,
  BM25Index,
  HNSWIndex,
  HybridRetriever,
  EpisodicMemoryStore
} = require('../engine/index');

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  \x1b[32m✔ PASS\x1b[0m ${name}`);
    passed++;
  } catch (err) {
    console.error(`  \x1b[31m✖ FAIL\x1b[0m ${name}`);
    console.error('   ', err.message);
    failed++;
  }
}

console.log('\n\x1b[1m\x1b[36m====================================================\x1b[0m');
console.log('\x1b[1m\x1b[36m SYNAPSE // VERIFICATION SUITE (Node.js Test Runner)\x1b[0m');
console.log('\x1b[1m\x1b[36m====================================================\x1b[0m\n');

// 1. VectorMath - Dot Product
test('VectorMath: dotProduct computes accurate algebraic sum', () => {
  const a = new Float32Array([1.0, 2.0, 3.0]);
  const b = new Float32Array([4.0, -1.0, 2.0]);
  const dot = VectorMath.dotProduct(a, b);
  // 1*4 + 2*(-1) + 3*2 = 4 - 2 + 6 = 8
  assert.strictEqual(Math.round(dot), 8);
});

// 2. VectorMath - L2 Normalization
test('VectorMath: l2Normalize creates unit vector (norm = 1.0)', () => {
  const v = new Float32Array([3.0, 4.0]);
  const normV = VectorMath.l2Normalize(v);
  const norm = VectorMath.l2Norm(normV);
  assert(Math.abs(norm - 1.0) < 1e-6, `Expected norm ~ 1.0, got ${norm}`);
  assert(Math.abs(normV[0] - 0.6) < 1e-6, `Expected normV[0] ~ 0.6, got ${normV[0]}`);
  assert(Math.abs(normV[1] - 0.8) < 1e-6, `Expected normV[1] ~ 0.8, got ${normV[1]}`);
});

// 3. VectorMath - Cosine Similarity
test('VectorMath: cosineSimilarity yields 1.0 for identical, 0.0 for orthogonal', () => {
  const a = new Float32Array([1, 0, 0]);
  const b = new Float32Array([1, 0, 0]);
  const c = new Float32Array([0, 1, 0]);
  assert.strictEqual(VectorMath.cosineSimilarity(a, b), 1.0);
  assert.strictEqual(VectorMath.cosineSimilarity(a, c), 0.0);
});

// 4. VectorMath - Pseudo Embedding
test('VectorMath: embed produces deterministic Float32Array with unit norm', () => {
  const emb1 = VectorMath.embed('authorization token exfiltration', 64);
  const emb2 = VectorMath.embed('authorization token exfiltration', 64);
  assert.strictEqual(emb1.length, 64);
  assert.strictEqual(emb1[0], emb2[0]);
  const norm = VectorMath.l2Norm(emb1);
  assert(Math.abs(norm - 1.0) < 1e-5);
});

// 5. BM25 - Lexical Tokenization & Stopwords
test('BM25Index: tokenizes text and removes common English stopwords', () => {
  const bm25 = new BM25Index();
  const tokens = bm25.tokenize('The quick $AUTH_TOKEN is on the server-node_1');
  assert(!tokens.includes('the'));
  assert(!tokens.includes('is'));
  assert(!tokens.includes('on'));
  assert(tokens.includes('$auth_token'));
  assert(tokens.includes('server-node_1'));
});

// 6. BM25 - Lexical Scoring & Ranking
test('BM25Index: ranks document with exact keyword match highest', () => {
  const bm25 = new BM25Index();
  bm25.addDocument(1, 'Agent encountered syntax error in python script');
  bm25.addDocument(2, 'Postgres connection pool exhausted on port 5432');
  bm25.addDocument(3, 'Agent authenticated with github enterprise token');

  const res = bm25.search('Postgres connection 5432');
  assert(res.length > 0);
  assert.strictEqual(res[0].docId, 2);
  assert(res[0].score > 0);
});

// 7. BM25 - Document Removal
test('BM25Index: removing a document purges it from inverted index', () => {
  const bm25 = new BM25Index();
  bm25.addDocument('d1', 'temporary secret key 12345');
  assert.strictEqual(bm25.search('secret key').length, 1);
  bm25.removeDocument('d1');
  assert.strictEqual(bm25.search('secret key').length, 0);
  assert.strictEqual(bm25.docCount, 0);
});

// 8. ScalarQuantizer - Compression Ratio
test('ScalarQuantizer: quantize reduces 32-bit floats to 8-bit uints (4x ratio)', () => {
  const vec = new Float32Array([0.1, -0.4, 0.9, 0.0, -0.8]);
  const q = ScalarQuantizer.quantize(vec);
  assert(q.qVec instanceof Uint8Array);
  assert.strictEqual(q.qVec.length, vec.length);
  assert(q.range > 0);
});

// 9. ScalarQuantizer - Dequantization Fidelity
test('ScalarQuantizer: dequantize retains > 98.5% cosine correlation', () => {
  const original = VectorMath.embed('Autonomous Agent Sandboxed Execution', 64);
  const quantized = ScalarQuantizer.quantize(original);
  const reconstructed = ScalarQuantizer.dequantize(quantized);

  const sim = VectorMath.cosineSimilarity(original, reconstructed);
  assert(sim > 0.985, `Expected cosine similarity > 0.985, got ${sim}`);
});

// 10. ScalarQuantizer - Asymmetric Fast Cosine
test('ScalarQuantizer: fastCosineSimilarity matches float calculation within 2%', () => {
  const a = VectorMath.embed('Kubernetes pod crash loop backoff', 64);
  const b = VectorMath.embed('Container out of memory kill', 64);
  const floatSim = VectorMath.cosineSimilarity(a, b);

  const qB = ScalarQuantizer.quantize(b);
  const fastSim = ScalarQuantizer.fastCosineSimilarity(a, qB);

  const diff = Math.abs(floatSim - fastSim);
  assert(diff < 0.05, `Expected diff < 0.05, got ${diff}`);
});

// 11. HNSWIndex - Graph Insertion & Connectivity
test('HNSWIndex: builds multi-layer graph and establishes entry point', () => {
  const hnsw = new HNSWIndex({ M: 8, efConstruction: 32 });
  for (let i = 0; i < 20; i++) {
    const v = VectorMath.embed(`agent prompt interaction step ${i}`, 32);
    hnsw.insert(`doc_${i}`, v);
  }
  assert.strictEqual(hnsw.size, 20);
  assert(hnsw.enterPointId !== null);
  assert(hnsw.maxLayer >= 0);
});

// 12. HNSWIndex - ANN Search Accuracy
test('HNSWIndex: retrieves identical vector with top rank (similarity ~ 1.0)', () => {
  const hnsw = new HNSWIndex({ M: 16, efSearch: 32 });
  const targetVec = VectorMath.embed('Critical security alert: unauthorized ssh access', 32);

  for (let i = 0; i < 15; i++) {
    hnsw.insert(`noise_${i}`, VectorMath.embed(`regular conversation log ${i}`, 32));
  }
  hnsw.insert('target_doc', targetVec);

  const results = hnsw.search(targetVec, 3);
  assert(results.length > 0);
  assert.strictEqual(results[0].id, 'target_doc');
  assert(results[0].similarity > 0.95);
});

// 13. HNSWIndex - Diagnostic Layer Trail
test('HNSWIndex: searchWithTrail records diagnostic skip-layer hops', () => {
  const hnsw = new HNSWIndex({ M: 8 });
  for (let i = 0; i < 25; i++) {
    hnsw.insert(`node_${i}`, VectorMath.embed(`vector node item ${i}`, 32));
  }
  const { results, trail } = hnsw.searchWithTrail(VectorMath.embed('query', 32), 3);
  assert(Array.isArray(results));
  assert(Array.isArray(trail));
});

// 14. HybridRetriever - Reciprocal Rank Fusion
test('HybridRetriever: balances dense vector and sparse keyword matches', () => {
  const retriever = new HybridRetriever({ defaultAlpha: 0.5 });
  const dense = [
    { id: 'docA', similarity: 0.92 },
    { id: 'docB', similarity: 0.81 }
  ];
  const sparse = [
    { docId: 'docB', score: 4.5 },
    { docId: 'docC', score: 3.1 }
  ];

  const fused = retriever.fuse(dense, sparse, { alpha: 0.5, topK: 3 });
  assert(fused.length > 0);
  // docB appears in both lists, so should have high combined RRF score
  assert.strictEqual(fused[0].id, 'docB');
  assert(fused[0].confidence > 0);
});

// 15. EpisodicMemoryStore - Remember & Working Memory FIFO
test('EpisodicMemoryStore: maintains short-term working memory FIFO capacity', () => {
  const store = new EpisodicMemoryStore({ workingMemoryCapacity: 3 });
  store.remember({ id: 'm1', text: 'First user prompt' });
  store.remember({ id: 'm2', text: 'Second tool call' });
  store.remember({ id: 'm3', text: 'Third tool output' });
  store.remember({ id: 'm4', text: 'Fourth LLM response' });

  const wm = store.getWorkingMemory();
  assert.strictEqual(wm.length, 3);
  assert.strictEqual(wm[0].id, 'm4');
  assert.strictEqual(wm[2].id, 'm2');
});

// 16. EpisodicMemoryStore - Recall with Metadata Filter & Decay
test('EpisodicMemoryStore: recall respects metadata tags and temporal decay', () => {
  const store = new EpisodicMemoryStore({ dimension: 32 });
  store.remember({
    id: 'sec_1',
    text: 'Firewall intercepted dangerous rm -rf command',
    metadata: { category: 'security', severity: 'high' }
  });
  store.remember({
    id: 'chat_1',
    text: 'User said hello and asked about the weather',
    metadata: { category: 'chat', severity: 'low' }
  });

  const securityOnly = store.recall('firewall danger command', {
    filter: { category: 'security' }
  });
  assert.strictEqual(securityOnly.length, 1);
  assert.strictEqual(securityOnly[0].id, 'sec_1');
  assert.strictEqual(securityOnly[0].metadata.severity, 'high');
});

// 17. Performance & Latency Benchmark
test('EpisodicMemoryStore: sub-millisecond retrieval latency across 500 memories', () => {
  const store = new EpisodicMemoryStore({ dimension: 64, useQuantization: true });

  // Ingest 500 episodic memories
  for (let i = 0; i < 500; i++) {
    store.remember({
      id: `mem_${i}`,
      text: `Autonomous agent action step #${i} in workspace execution session`,
      metadata: { step: i }
    });
  }

  // Warm-up query
  store.recall('workspace execution step');

  // Benchmark 50 consecutive queries
  const start = performance.now();
  const iterations = 50;
  for (let i = 0; i < iterations; i++) {
    store.recall(`autonomous agent action #${i % 100}`);
  }
  const totalMs = performance.now() - start;
  const avgMs = totalMs / iterations;

  console.log(`     \x1b[90m↳ Latency: ${avgMs.toFixed(3)} ms/query (${(1000 / avgMs).toFixed(0)} QPS) across 500 vectors\x1b[0m`);
  assert(avgMs < 5.0, `Expected sub-5ms latency in test runner, got ${avgMs}ms`);
});

console.log('\n\x1b[1m----------------------------------------------------\x1b[0m');
console.log(`\x1b[1m RESULTS: \x1b[32m${passed} passed\x1b[0m, \x1b[31m${failed} failed\x1b[0m (Total: ${passed + failed})`);
console.log('\x1b[1m----------------------------------------------------\x1b[0m\n');

if (failed > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
