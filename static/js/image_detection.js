/* VCare AI — Image Detection Script */

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewWrapper = document.getElementById('previewWrapper');
    const imagePreview = document.getElementById('imagePreview');
    const uploadActions = document.getElementById('uploadActions');
    const changeImageBtn = document.getElementById('changeImageBtn');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const resultEl = document.getElementById('resultContainer');

    // Sidebar
    const sidebar = document.getElementById('sidebar');
    document.getElementById('sidebarToggle')?.addEventListener('click', () => sidebar.classList.toggle('collapsed'));
    document.getElementById('mobileMenuBtn')?.addEventListener('click', () => sidebar.classList.toggle('open'));

    let selectedFile = null;

    // Browse button / zone click
    browseBtn.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', e => {
        const f = e.target.files[0];
        if (f) loadFile(f);
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', e => {
        e.preventDefault();
        uploadArea.classList.add('dragging');
    });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragging'));
    uploadArea.addEventListener('drop', e => {
        e.preventDefault();
        uploadArea.classList.remove('dragging');
        const f = e.dataTransfer.files[0];
        if (f && f.type.startsWith('image/')) loadFile(f);
    });

    function loadFile(file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            uploadArea.querySelector('.upload-zone-inner').style.display = 'none';
            previewWrapper.style.display = 'block';
            uploadActions.style.display = 'flex';
            resultEl.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // Change image
    changeImageBtn.addEventListener('click', resetUI);

    // Reset
    resetBtn.addEventListener('click', resetUI);

    function resetUI() {
        fileInput.value = '';
        selectedFile = null;
        imagePreview.src = '';
        uploadArea.querySelector('.upload-zone-inner').style.display = 'flex';
        previewWrapper.style.display = 'none';
        uploadActions.style.display = 'none';
        resultEl.style.display = 'none';
    }

    // Analyze
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        setLoading(true);

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const res = await fetch('/predict_skin_cancer', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error(data.error || 'Analysis failed');
            showResult(data);
        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(on) {
        predictBtn.disabled = on;
        predictBtn.innerHTML = on
            ? '<i class="fas fa-spinner"></i><span>Analyzing…</span>'
            : '<i class="fas fa-brain"></i><span>Analyze Image</span>';
    }

    function showResult(data) {
        const prob = parseFloat(data.probability) || 0;
        const diagnosis = data.diagnosis || 'Unknown';
        const description = data.description || '';

        let riskClass, riskIcon;
        if (prob > 70) { riskClass = 'high'; riskIcon = 'fa-circle-exclamation'; }
        else if (prob > 40) { riskClass = 'moderate'; riskIcon = 'fa-circle-exclamation'; }
        else { riskClass = 'low'; riskIcon = 'fa-circle-check'; }

        resultEl.innerHTML = `
            <div class="result-panel-title">
                <i class="fas fa-chart-bar"></i>
                Analysis Result
            </div>

            <div class="risk-badge ${riskClass}">
                <i class="fas ${riskIcon}"></i>
                ${diagnosis}
            </div>

            <div class="prob-bar-wrapper">
                <div class="prob-bar-label">
                    <span>Cancer Risk Probability</span>
                    <span>${prob.toFixed(2)}%</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" id="probFill" style="width: 0%"></div>
                </div>
            </div>

            <div class="result-description">
                ${description}
            </div>

            <div class="result-disclaimer">
                <i class="fas fa-triangle-exclamation"></i>
                This AI analysis is for <strong>screening purposes only</strong>. Always consult a qualified dermatologist or oncologist for a confirmed clinical diagnosis.
            </div>
        `;

        resultEl.style.display = 'block';

        setTimeout(() => {
            const fill = document.getElementById('probFill');
            if (fill) fill.style.width = prob.toFixed(2) + '%';
        }, 80);

        resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showError(msg) {
        resultEl.innerHTML = `
            <div class="result-panel-title">
                <i class="fas fa-xmark-circle" style="color: var(--red-400)"></i>
                Error
            </div>
            <div class="result-description" style="border-left-color: var(--red-600);">
                ${msg}
            </div>
        `;
        resultEl.style.display = 'block';
    }
});
