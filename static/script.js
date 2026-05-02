let selectedFile = null;

const dropZone    = document.getElementById('dropZone');
const fileInput   = document.getElementById('fileInput');
const previewZone = document.getElementById('previewZone');
const previewImg  = document.getElementById('previewImg');
const fileName    = document.getElementById('fileName');
const analyzeBtn  = document.getElementById('analyzeBtn');
const loading     = document.getElementById('loading');
const results     = document.getElementById('resultsSection');

// Drag & Drop
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src   = e.target.result;
    fileName.textContent = file.name;
    dropZone.style.display   = 'none';
    previewZone.style.display = 'block';
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  dropZone.style.display    = 'block';
  previewZone.style.display = 'none';
  analyzeBtn.disabled = true;
}

function resetAll() {
  resetUpload();
  results.style.display = 'none';
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Color per class
const classColors = {
  glioma:     '#ff5555',
  meningioma: '#ffaa00',
  notumor:    '#00dd88',
  pituitary:  '#4499ff'
};

async function analyzeImage() {
  if (!selectedFile) return;

  loading.style.display  = 'flex';
  results.style.display  = 'none';
  analyzeBtn.disabled    = true;

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res  = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (data.error) {
      alert('Error: ' + data.error);
      loading.style.display = 'none';
      analyzeBtn.disabled   = false;
      return;
    }

    showResults(data);

  } catch (err) {
    alert('Server se connection nahi ho saka. app.py chal raha hai?');
    loading.style.display = 'none';
    analyzeBtn.disabled   = false;
  }
}

function showResults(data) {
  loading.style.display = 'none';
  results.style.display = 'block';

  const color = classColors[data.class] || '#00e5ff';

  // Diagnosis
  document.getElementById('diagnosisName').textContent = data.class;
  document.getElementById('diagnosisName').style.color = color;
  document.getElementById('confValue').textContent     = data.confidence.toFixed(1) + '%';
  document.getElementById('confValue').style.color     = color;
  document.getElementById('diagnosisDesc').textContent = data.description;
  document.getElementById('adviceText').textContent    = data.advice;

  // Confidence bar
  setTimeout(() => {
    document.getElementById('confBar').style.width      = data.confidence + '%';
    document.getElementById('confBar').style.background = color;
  }, 100);

  // Probability bars
  const probContainer = document.getElementById('probBars');
  probContainer.innerHTML = '';
  const sorted = Object.entries(data.all_probs).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cls, prob]) => {
    const c   = classColors[cls] || '#00e5ff';
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <div class="prob-row-label">
        <span>${cls}</span>
        <span>${prob.toFixed(1)}%</span>
      </div>
      <div class="prob-row-bar">
        <div class="prob-row-fill" style="width:0%; background:${c}"></div>
      </div>`;
    probContainer.appendChild(row);
    setTimeout(() => {
      row.querySelector('.prob-row-fill').style.width = prob + '%';
    }, 150);
  });

  // Images
  document.getElementById('originalImg').src  = 'data:image/png;base64,' + data.original;
  document.getElementById('annotatedImg').src = 'data:image/png;base64,' + data.annotated;

  results.scrollIntoView({ behavior: 'smooth' });
}
