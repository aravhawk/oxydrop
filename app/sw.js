const V = 'oxydrop-v3.4-ios-frames';
// everything the hybrid needs offline: the page, the engines, the CNN worker, the three CNN seed files, the physics-chain/meta/conformal parameters and the replay demo
const ASSETS = ['./', 'index.html', 'style.css', 'manifest.webmanifest', 'icons/icon.svg',
  'js/app.js', 'js/decision.js', 'js/camera.js', 'js/resample.js', 'js/engine.js', 'js/engine_v1.js', 'js/hr.js', 'js/lensprobe.js',
  'js/hybrid_engine.js', 'js/hybrid_async.js', 'js/hybrid_worker.js', 'js/cnn.js', 'js/cnn_features.js',
  'data/weights.json', 'data/hybrid_weights.json', 'data/conformal.json', 'data/hybrid_R_s0.json', 'data/hybrid_R_s1.json', 'data/hybrid_R_s2.json',
  'data/replay/100001L.json', 'data/replay/100001L.bin', 'data/replay/100004R.json', 'data/replay/100004R.bin'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(V).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== V).map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
// network-first so a deployed update is never hidden by a stale cached script; the cache is only the offline fallback
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(fetch(e.request).then((r) => { const cp = r.clone(); caches.open(V).then((c) => c.put(e.request, cp)); return r; }).catch(() => caches.match(e.request)));
});
