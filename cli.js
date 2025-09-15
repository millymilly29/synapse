#!/usr/bin/env node

/**
 * SYNAPSE // CLI Interface & Benchmarking Tool
 * Zero-dependency In-Memory Vector Database & Hybrid Search Engine
 */

const { EpisodicMemoryStore } = require('./engine/memory-store');
const { VectorMath } = require('./engine/vector-math');
const { ScalarQuantizer } = require('./engine/quantizer');

const SEED_MEMORIES = [
  { id: 'MEM-001', text: 'Agent intercepted suspicious AWS IAM token leak in environment dump', metadata: { category: 'security', priority: 'critical' } },
  { id: 'MEM-002', text: 'Executed docker system prune -a to reclaim 18.4GB disk space', metadata: { category: 'devops', priority: 'medium' } },
  { id: 'MEM-003', text: 'User requested financial revenue export for Q3 2026 in CSV format', metadata: { category: 'user_intent', priority: 'high' } },
  { id: 'MEM-004', text: 'PostgreSQL connection timeout on port 5432 after 3000ms idle', metadata: { category: 'database', priority: 'high' } },
  { id: 'MEM-005', text: 'Firewall blocked curl payload piped to /bin/bash from untrusted host', metadata: { category: 'security', priority: 'critical' } },
  { id: 'MEM-006', text: 'Refactored React tree into lazy dynamic Suspense boundaries', metadata: { category: 'frontend', priority: 'low' } },
  { id: 'MEM-007', text: 'JWT authentication token expired for session uuid-4982a-bc01', metadata: { category: 'auth', priority: 'medium' } },
  { id: 'MEM-008', text: 'Kubernetes node memory pressure triggered OOM killer on worker-04', metadata: { category: 'devops', priority: 'high' } }
];

function printBanner() {
  console.log(`
\x1b[36m   ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ███████╗███████╗
   ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝
   ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝███████╗█████╗  
   ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝ ╚════██║██╔══╝  
   ███████║   ██║   ██║ ╚████║██║  ██║██║     ███████║███████╗
   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝\x1b[0m
 \x1b[90m// SUB-MILLISECOND IN-MEMORY VECTOR & HYBRID RETRIEVAL ENGINE\x1b[0m
`);
}

function runDemo() {
  printBanner();
  console.log('\x1b[1m\x1b[32m[INIT]\x1b[0m Initializing EpisodicMemoryStore (HNSW + BM25 + SQ8 Quantization)...');
  const store = new EpisodicMemoryStore({ dimension: 64, useQuantization: true });

  console.log('\x1b[1m\x1b[34m[SEED]\x1b[0m Ingesting 8 baseline autonomous agent episodic memories...');
  SEED_MEMORIES.forEach(m => store.remember(m));

  const telemetry = store.getTelemetry();
  console.log(`\x1b[1m\x1b[35m[TELEMETRY]\x1b[0m Dimension: ${telemetry.dimension}d | Records: ${telemetry.totalMemories} | Compression: ${telemetry.compressionRatio} (${telemetry.bytesSaved} bytes saved)\n`);

  const testQueries = [
    { q: 'credential exfiltration and token leakage', label: 'Semantic General Search' },
    { q: 'docker prune reclaim', label: 'Exact Keyword Match' },
    { q: 'postgres connection timeout 5432', label: 'Mixed Hybrid Technical Diagnostic' }
  ];

  for (const { q, label } of testQueries) {
    console.log(`\x1b[1m--------------------------------------------------------------\x1b[0m`);
    console.log(`\x1b[33mQUERY:\x1b[0m "${q}" \x1b[90m(${label})\x1b[0m`);
    const start = performance.now();
    const results = store.recall(q, { topK: 3, alpha: 0.65 });
    const elapsed = (performance.now() - start).toFixed(3);

    console.log(`\x1b[90mExecution time: ${elapsed} ms\x1b[0m`);
    results.forEach((r, idx) => {
      console.log(`  \x1b[36m#${idx + 1}\x1b[0m [\x1b[32m${(r.confidence * 100).toFixed(1)}%\x1b[0m confidence] [${r.id}] ${r.text}`);
      console.log(`     \x1b[90m↳ Vector Sim: ${r.vectorSim} | BM25: ${r.bm25Score} | Cat: ${r.metadata.category}\x1b[0m`);
    });
    console.log();
  }
}

function runBenchmark(count = 2000, dim = 64) {
  printBanner();
  console.log(`\x1b[1m\x1b[33m[BENCHMARK]\x1b[0m Stress testing with ${count} synthetic vectors (${dim}-dimensional)...`);

  const store = new EpisodicMemoryStore({ dimension: dim, useQuantization: true });

  // 1. Insertion Benchmark
  const t0 = performance.now();
  for (let i = 0; i < count; i++) {
    const text = `Agent telemetry log item #${i}: system process PID-${1000 + i} status normal memory OK`;
    store.remember({
      id: `DOC-${i}`,
      text,
      metadata: { shard: i % 4, pid: 1000 + i }
    });
  }
  const insertDurationMs = performance.now() - t0;
  const insertThroughput = Math.round((count / (insertDurationMs / 1000)));

  // 2. Query Latency Benchmark
  const queryCount = 200;
  const latencies = [];

  for (let i = 0; i < queryCount; i++) {
    const query = `system process telemetry status #${i % count}`;
    const qStart = performance.now();
    store.recall(query, { topK: 5, alpha: 0.65 });
    latencies.push(performance.now() - qStart);
  }

  latencies.sort((a, b) => a - b);
  const p50 = latencies[Math.floor(queryCount * 0.50)].toFixed(3);
  const p90 = latencies[Math.floor(queryCount * 0.90)].toFixed(3);
  const p95 = latencies[Math.floor(queryCount * 0.95)].toFixed(3);
  const p99 = latencies[Math.floor(queryCount * 0.99)].toFixed(3);
  const avg = (latencies.reduce((a, b) => a + b, 0) / queryCount).toFixed(3);
  const qps = Math.round(1000 / avg);

  const tel = store.getTelemetry();

  console.log('\n\x1b[1m\x1b[32m=== BENCHMARK RESULTS ===\x1b[0m');
  console.log(`  Indexed Records:       ${count}`);
  console.log(`  Vector Dimension:      ${dim}d`);
  console.log(`  Insert Throughput:     \x1b[36m${insertThroughput.toLocaleString()} vectors/sec\x1b[0m (Total: ${insertDurationMs.toFixed(1)}ms)`);
  console.log(`  Query Throughput:      \x1b[36m${qps.toLocaleString()} QPS\x1b[0m`);
  console.log(`  P50 Latency:           \x1b[32m${p50} ms\x1b[0m`);
  console.log(`  P90 Latency:           \x1b[32m${p90} ms\x1b[0m`);
  console.log(`  P95 Latency:           \x1b[33m${p95} ms\x1b[0m`);
  console.log(`  P99 Latency:           \x1b[33m${p99} ms\x1b[0m`);
  console.log(`  Memory (Raw Float32):  ${(tel.rawVectorBytes / 1024).toFixed(1)} KB`);
  console.log(`  Memory (SQ8 Quant):    ${(tel.activeStorageBytes / 1024).toFixed(1)} KB`);
  console.log(`  Compression Factor:    \x1b[35m${tel.compressionRatio} (75% savings)\x1b[0m`);
  console.log('=========================\n');
}

function handleSearch(args) {
  const queryStr = args.find(a => !a.startsWith('--'));
  if (!queryStr) {
    console.error('\x1b[31mError:\x1b[0m Please provide a query string. Example: node cli.js search "auth token"');
    process.exit(1);
  }

  const alphaArg = args.find(a => a.startsWith('--alpha='));
  const alpha = alphaArg ? parseFloat(alphaArg.split('=')[1]) : 0.65;

  const topArg = args.find(a => a.startsWith('--top='));
  const topK = topArg ? parseInt(topArg.split('=')[1], 10) : 5;

  const store = new EpisodicMemoryStore({ dimension: 64, useQuantization: true });
  SEED_MEMORIES.forEach(m => store.remember(m));

  console.log(`\n\x1b[1m\x1b[33m[SYNAPSE SEARCH]\x1b[0m Query: "${queryStr}" (Alpha: ${alpha}, TopK: ${topK})`);
  const t0 = performance.now();
  const results = store.recall(queryStr, { topK, alpha });
  const ms = (performance.now() - t0).toFixed(3);

  console.log(`\x1b[90mRetrieved in ${ms} ms\x1b[0m\n`);
  if (results.length === 0) {
    console.log('  No matching memories found.');
  } else {
    results.forEach((r, idx) => {
      console.log(`  \x1b[36m#${idx + 1}\x1b[0m [\x1b[32m${(r.confidence * 100).toFixed(1)}%\x1b[0m] ${r.text}`);
      console.log(`     \x1b[90mVector: ${r.vectorSim} | BM25: ${r.bm25Score} | Metadata: ${JSON.stringify(r.metadata)}\x1b[0m`);
    });
  }
  console.log();
}

// Main CLI router
const args = process.argv.slice(2);
const command = args[0];

if (!command || command === 'demo') {
  runDemo();
} else if (command === 'benchmark') {
  const countArg = args.find(a => a.startsWith('--count='));
  const count = countArg ? parseInt(countArg.split('=')[1], 10) : 2000;
  const dimArg = args.find(a => a.startsWith('--dim='));
  const dim = dimArg ? parseInt(dimArg.split('=')[1], 10) : 64;
  runBenchmark(count, dim);
} else if (command === 'search') {
  handleSearch(args.slice(1));
} else if (command === '--help' || command === '-h' || command === 'help') {
  printBanner();
  console.log(`Usage:
  node cli.js                     Run interactive demo with seed memories
  node cli.js benchmark [opts]    Run high-throughput vector benchmark
     --count=2000                 Number of vectors to benchmark (default: 2000)
     --dim=64                     Vector dimensions (default: 64)
  node cli.js search "<query>"    Search seed agent memories
     --alpha=0.65                 Dense/sparse blend (0.0=BM25, 1.0=Vector)
     --top=5                      Number of results (default: 5)
  node tests/synapse.test.js      Run 17-point test suite
`);
} else {
  console.error(`\x1b[31mUnknown command:\x1b[0m ${command}. Run "node cli.js --help" for options.`);
  process.exit(1);
}
