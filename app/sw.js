const V = 'oxydrop-v1';
const ASSETS = ['./', 'index.html', 'style.css', 'js/app.js', 'js/engine.js', 'js/camera.js', 'data/weights.json', 'manifest.webmanifest', 'icons/icon.svg',
  'data/replay/100001L.json', 'data/replay/100001L.bin', 'data/replay/100004R.json', 'data/replay/100004R.bin'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(V).then((c) => c.addAll(ASSETS)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== V).map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(fetch(e.request).then((r) => { const cp = r.clone(); caches.open(V).then((c) => c.put(e.request, cp)); return r; }).catch(() => caches.match(e.request)));
});
