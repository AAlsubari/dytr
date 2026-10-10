// ==============================================================================
// DYTR Studio Standalone Mock API Server (Zero External Dependencies)
// Strictly decouples frontend from static arrays and simulates real-world REST API
// ==============================================================================

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = process.env.MOCK_SERVER_PORT || 3001;
const DB_PATH = path.join(__dirname, 'db.json');

function loadDb() {
  try {
    const raw = fs.readFileSync(DB_PATH, 'utf-8');
    return JSON.parse(raw);
  } catch (err) {
    console.error('Failed to read db.json:', err.message);
    return {};
  }
}

const server = http.createServer((req, res) => {
  // Global CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    return res.end();
  }

  // Realistic network latency simulation (60ms - 150ms)
  const delay = Math.floor(Math.random() * 90) + 60;

  setTimeout(() => {
    try {
      const db = loadDb();
      const parsedUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
      const pathname = parsedUrl.pathname;

      if (pathname === '/api/v1/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify({ status: 'ok', server: 'DYTR Studio Mock Server', port: PORT }));
      }

      if (pathname === '/api/v1/problem/scenarios' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(db.problem_scenarios || []));
      }

      if (pathname === '/api/v1/problem/diagnostic-questions' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(db.diagnostic_questions || []));
      }

      if (pathname === '/api/v1/problem/task-types' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(db.task_types || []));
      }

      if (pathname === '/api/v1/problem/foundation-models' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(db.foundation_models || []));
      }

      if (pathname === '/api/v1/datasets/catalog' && req.method === 'GET') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        return res.end(JSON.stringify(db.sample_datasets_catalog || []));
      }

      // 404 for unmatched routes
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'error', message: `Route ${pathname} not found on Mock Server` }));
    } catch (e) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'error', message: e.message }));
    }
  }, delay);
});

server.listen(PORT, () => {
  console.log(`=======================================================`);
  console.log(`  DYTR Studio Mock API Server running on port ${PORT}`);
  console.log(`  Base Endpoint: http://localhost:${PORT}/api/v1`);
  console.log(`=======================================================`);
});
