/* VCare AI — Advanced Blood Sample Analysis */

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('bloodSampleForm');
    const predictBtn = document.getElementById('predictBtn');
    const resultEl = document.getElementById('resultContainer');
    const resetBtn = document.getElementById('resetBtn');

    // Sidebar
    const sidebar = document.getElementById('sidebar');
    document.getElementById('sidebarToggle')?.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });
    document.getElementById('mobileMenuBtn')?.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        setLoading(true);
        resultEl.style.display = 'none';

        // Extract 16 features in correct order: Gender, Age, Hb, RBC, WBC, PLATELETS, LYMP, MONO, HCT, MCV, MCH, MCHC, RDW, PDW, MPV, PCT
        const data = {
            gender: parseInt(form.gender.value),
            age: parseFloat(form.age.value),
            hb: parseFloat(form.hb.value),
            rbc: parseFloat(form.rbc.value),
            wbc: parseFloat(form.wbc.value),
            platelets: parseFloat(form.platelets.value),
            lymp: parseFloat(form.lymp.value),
            mono: parseFloat(form.mono.value),
            hct: parseFloat(form.hct.value),
            mcv: parseFloat(form.mcv.value),
            mch: parseFloat(form.mch.value),
            mchc: parseFloat(form.mchc.value),
            rdw: parseFloat(form.rdw.value),
            pdw: parseFloat(form.pdw.value),
            mpv: parseFloat(form.mpv.value),
            pct: parseFloat(form.pct.value)
        };

        try {
            const res = await fetch('/predict_blood_sample', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await res.json();
            
            if (!res.ok || !result.success) {
                throw new Error(result.error || 'Prediction failed');
            }
            
            showResult(result);

        } catch (err) {
            console.error('Prediction error:', err);
            showError(err.message);
        } finally {
            setLoading(false);
        }
    });

    // Reset button
    resetBtn.addEventListener('click', () => {
        resultEl.style.display = 'none';
    });

    function setLoading(on) {
        predictBtn.disabled = on;
        if (on) {
            predictBtn.classList.add('loading');
            predictBtn.innerHTML = '<i class="fas fa-spinner"></i><span>Analyzing…</span><span class="btn-subtext">Processing data</span>';
        } else {
            predictBtn.classList.remove('loading');
            predictBtn.innerHTML = '<i class="fas fa-wand-magic-sparkles"></i><span>Run Advanced Analysis</span><span class="btn-subtext">AI-Powered Prediction</span>';
        }
    }

    function showResult(data) {
        const isPositive = data.raw_prediction === 1;
        const prob = data.probability !== null ? (data.probability * 100).toFixed(1) : null;
        const confidence = prob !== null ? `${prob}%` : 'N/A';
        const riskClass = isPositive ? 'high' : 'low';
        const riskIcon = isPositive ? 'fa-circle-exclamation' : 'fa-circle-check';
        const timestamp = new Date().toLocaleString();

        let resultHTML = `
            <div class="result-panel-title">
                <i class="fas fa-chart-bar"></i>
                Advanced Analysis Result
            </div>

            <div class="risk-badge ${riskClass}">
                <i class="fas ${riskIcon}"></i>
                ${data.prediction}
            </div>

            <div class="result-details">
                <div class="detail-item">
                    <span class="detail-label">Analysis Type:</span>
                    <span class="detail-value">CatBoost Classification Model</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Timestamp:</span>
                    <span class="detail-value">${timestamp}</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Raw Prediction:</span>
                    <span class="detail-value">${isPositive ? 'Positive (1)' : 'Negative (0)'}</span>
                </div>
            </div>

            ${prob !== null ? `
            <div class="prob-bar-wrapper">
                <div class="prob-bar-label">
                    <span>Confidence Score</span>
                    <span class="confidence-badge">${confidence}</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" id="probFill" style="width: 0%"></div>
                </div>
            </div>
            ` : ''}

            <div class="result-description">
                ${isPositive
                ? '<strong>High Risk Indicators Detected:</strong> The analysis identified elevated bone marrow blasts, abnormal blood cell counts, or other concerning markers. <strong>Immediate medical consultation is recommended.</strong>'
                : '<strong>Normal Risk Profile:</strong> Blood parameters are within normal ranges. No strong indicators of leukemia were detected based on the provided test data.'}
            </div>

            <div class="result-actions">
                <button class="btn-action" onclick="window.print()">
                    <i class="fas fa-print"></i> Print Report
                </button>
                <button class="btn-action" onclick="downloadReport('${JSON.stringify(data).replace(/'/g, "\\'")}')">
                    <i class="fas fa-download"></i> Download Report
                </button>
            </div>

            <div class="result-disclaimer">
                <i class="fas fa-triangle-exclamation"></i>
                <strong>IMPORTANT DISCLAIMER:</strong> This AI analysis is intended for <strong>informational purposes only</strong> and should <strong>NOT</strong> be used for clinical diagnosis. Please consult a licensed hematologist or oncologist for medical diagnosis and treatment recommendations.
            </div>
        `;

        resultEl.innerHTML = resultHTML;
        resultEl.style.display = 'block';

        // Animate confidence bar
        if (prob !== null) {
            setTimeout(() => {
                const fill = document.getElementById('probFill');
                if (fill) fill.style.width = confidence;
            }, 100);
        }

        resultEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showError(msg) {
        resultEl.innerHTML = `
            <div class="result-panel-title">
                <i class="fas fa-exclamation-circle" style="color: #dc2626"></i>
                Analysis Error
            </div>
            <div class="result-description error-description">
                <strong>Error:</strong> ${msg}
            </div>
            <p style="font-size: 13px; color: #6b7280; margin-top: 12px;">
                Please check your input values and try again. If the problem persists, contact support.
            </p>
        `;
        resultEl.style.display = 'block';
        resultEl.scrollIntoView({ behavior: 'smooth' });
    }

    // Download report function
    window.downloadReport = function(data) {
        const reportContent = `
Blood Sample Analysis Report
============================
Generated: ${new Date().toLocaleString()}

ANALYSIS DATA:
${data}

DISCLAIMER:
This report is for informational purposes only and should not be used for clinical diagnosis.
Consult a licensed healthcare professional for medical advice.
        `;
        
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `blood-analysis-${Date.now()}.txt`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    };
});

        const prob = data.probability !== null ? (data.probability * 100).toFixed(1) : null;
        const confidence = prob !== null ? `${prob}%` : 'N/A';
        const riskClass = isPositive ? 'high' : 'low';
        const riskIcon = isPositive ? 'fa-circle-exclamation' : 'fa-circle-check';

        resultEl.innerHTML = `
            <div class="result-panel-title">
                <i class="fas fa-chart-bar"></i>
                Prediction Result
            </div>

            <div class="risk-badge ${riskClass}">
                <i class="fas ${riskIcon}"></i>
                ${data.prediction}
            </div>

            ${prob !== null ? `
            <div class="prob-bar-wrapper">
                <div class="prob-bar-label">
                    <span>Confidence Score</span>
                    <span>${confidence}</span>
                </div>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" id="probFill" style="width: 0%"></div>
                </div>
            </div>
            ` : ''}

            <div class="result-description">
                ${isPositive
                ? 'The model detected indicators consistent with <strong>leukemia risk</strong>. Elevated bone marrow blast percentages, abnormal blood cell counts, or combination of risk factors may have contributed to this prediction.'
                : 'The blood sample parameters fall within <strong>normal or low-risk ranges</strong>. No strong indicators of leukemia were detected based on the provided data.'}
            </div>

            <div class="result-disclaimer">
                <i class="fas fa-triangle-exclamation"></i>
                This prediction is generated by an AI model and is intended for <strong>informational purposes only</strong>. Please consult a licensed hematologist or oncologist for clinical diagnosis and treatment.
            </div>
        `;

        resultEl.style.display = 'block';

        // Animate confidence bar
        if (prob !== null) {
            setTimeout(() => {
                const fill = document.getElementById('probFill');
                if (fill) fill.style.width = confidence;
            }, 80);
        }

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
        resultEl.scrollIntoView({ behavior: 'smooth' });
    }
});
