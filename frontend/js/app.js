// ══════════════════════════════════════════════════════════
//  Brain Tumor Detection — Frontend JavaScript
// ══════════════════════════════════════════════════════════

// ┌─────────────────────────────────────────────────────────┐
// │  API_BASE: Empty = same origin (local dev)              │
// │  For Vercel+ngrok: set to your ngrok URL                │
// │  Example: "https://xxxx.ngrok-free.app"                 │
// └─────────────────────────────────────────────────────────┘
const API_BASE = "";

const icons = { glioma: '🔴', meningioma: '🟠', notumor: '🟢', pituitary: '🟡' };
const labels = {
  glioma: 'Glioma Tumor', meningioma: 'Meningioma Tumor',
  notumor: 'No Tumor (Healthy)', pituitary: 'Pituitary Tumor'
};

const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag') });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('drag');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleFile(fileInput.files[0]) });

let selectedFile = null;
function handleFile(f) {
  selectedFile = f;
  const r = new FileReader();
  r.onload = e => {
    document.getElementById('prev-img').src = e.target.result;
    document.getElementById('prev-name').textContent = f.name + ' (' + (f.size / 1024).toFixed(1) + ' KB)';
    document.getElementById('preview-box').style.display = 'block';
  };
  r.readAsDataURL(f);
  document.getElementById('analyze-btn').disabled = false;
  hideErr();
  document.getElementById('results').style.display = 'none';
  document.getElementById('placeholder').style.display = 'block';
  document.getElementById('chatbot-fab').style.display = 'none';
  document.getElementById('chatbot-popup').classList.remove('active');
  chatHistory = [];
  document.getElementById('chat-msgs').innerHTML = '<div class="msg bot">Hi! I am medical assistant. How can I help you understand your results?</div>';
}

function analyze() {
  if (!selectedFile) return;
  const btn = document.getElementById('analyze-btn');
  btn.classList.add('loading'); btn.disabled = true; hideErr();
  const fd = new FormData(); fd.append('file', selectedFile);
  fetch(`${API_BASE}/predict`, { method: 'POST', body: fd })
    .then(r => r.json()).then(d => {
      btn.classList.remove('loading'); btn.disabled = false;
      if (d.error) { showErr(d.error); return; }
      showResults(d);
      addSuggestedChip(d.class);
    }).catch(e => { btn.classList.remove('loading'); btn.disabled = false; showErr(e.message) });
}

function showResults(d) {
  document.getElementById('placeholder').style.display = 'none';
  document.getElementById('results').style.display = 'block';
  document.getElementById('orig-img').src = 'data:image/png;base64,' + d.original;
  document.getElementById('ann-img').src = 'data:image/png;base64,' + d.annotated;
  const banner = document.getElementById('diag-banner');
  banner.className = 'diag-banner ' + d.class;
  document.getElementById('diag-icon').textContent = icons[d.class] || '⚪';
  document.getElementById('diag-class').textContent = labels[d.class] || d.class;
  const cb = document.getElementById('diag-conf');
  cb.textContent = d.confidence.toFixed(1) + '%';
  cb.className = d.confidence >= 80 ? 'conf-hi' : d.confidence >= 50 ? 'conf-mid' : 'conf-lo';
  document.getElementById('r-desc').textContent = d.description;
  document.getElementById('r-advice').textContent = d.advice;
  const pb = document.getElementById('prob-bars'); pb.innerHTML = '';
  Object.entries(d.all_probs).sort((a, b) => b[1] - a[1]).forEach(([cls, pct]) => {
    pb.innerHTML += `<div class="prob-row"><div class="prob-hdr"><span class="prob-name">${icons[cls] || ''} ${cls}</span><span class="prob-val">${pct.toFixed(1)}%</span></div><div class="prob-bg"><div class="prob-bar bar-${cls}" style="width:${pct}%"></div></div></div>`;
  });
}

function showErr(m) { const b = document.getElementById('err-box'); b.style.display = 'block'; b.textContent = '⚠️ ' + m }
function hideErr() { const b = document.getElementById('err-box'); b.style.display = 'none'; b.textContent = '' }

// ══════════════════════════════════════════════════════════
//  CHATBOT LOGIC
// ══════════════════════════════════════════════════════════
let chatHistory = [];

function addMsg(text, role) {
  const box = document.getElementById('chat-msgs');
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
  const now = new Date();
  const time = now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0');
  div.innerHTML = text.replace(/\n/g, '<br>') + `<div class="msg-time">${time}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

function addTyping() {
  const div = addMsg('Typing...', 'bot');
  div.classList.add('typing');
  div.id = 'typing-indicator';
  return div;
}

function removeTyping() {
  const t = document.getElementById('typing-indicator');
  if (t) t.remove();
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const text = input.value.trim();
  if (!text) return;
  addMsg(text, 'user');
  chatHistory.push({ role: 'user', content: text });
  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;
  addTyping();
  try {
    const resp = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, history: chatHistory.slice(-10) })
    });
    const data = await resp.json();
    removeTyping();
    if (data.error) {
      addMsg('⚠️ Error: ' + data.error, 'bot');
    } else {
      addMsg(data.reply, 'bot');
      chatHistory.push({ role: 'assistant', content: data.reply });
    }
  } catch (e) {
    removeTyping();
    addMsg('⚠️ Could not connect to server.', 'bot');
  }
  sendBtn.disabled = false;
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function chipClick(el) {
  document.getElementById('chat-input').value = el.textContent;
  sendMessage();
}

function toggleChat() {
  const popup = document.getElementById('chatbot-popup');
  popup.classList.toggle('active');
  if (popup.classList.contains('active')) document.getElementById('chat-input').focus();
}

function addSuggestedChip(cls) {
  const qmap = {
    glioma: 'What is the treatment for Glioma?',
    meningioma: 'What happens after Meningioma surgery?',
    pituitary: 'What are Pituitary tumor symptoms?',
    notumor: 'What to do after a normal MRI result?'
  };
  const chips = document.getElementById('chips');
  const q = qmap[cls];
  if (!q) return;
  document.getElementById('chatbot-fab').style.display = 'flex';
  const existing = [...chips.querySelectorAll('.chip')].find(c => c.textContent === q);
  if (existing) return;
  const chip = document.createElement('div');
  chip.className = 'chip';
  chip.textContent = q;
  chip.onclick = () => chipClick(chip);
  chip.style.borderColor = 'rgba(155,89,182,.4)';
  chip.style.color = '#c39bd3';
  chips.prepend(chip);
}
