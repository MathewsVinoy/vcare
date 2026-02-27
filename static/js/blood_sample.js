document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('bloodSampleForm');
    const resultContainer = document.getElementById('resultContainer');
    const resultContent = document.getElementById('resultContent');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Get form data
        const formData = {
            age: parseFloat(document.getElementById('age').value),
            gender: parseInt(document.getElementById('gender').value),
            wbc_count: parseFloat(document.getElementById('wbc_count').value),
            rbc_count: parseFloat(document.getElementById('rbc_count').value),
            platelet_count: parseFloat(document.getElementById('platelet_count').value),
            hemoglobin_level: parseFloat(document.getElementById('hemoglobin_level').value),
            bone_marrow_blasts: parseFloat(document.getElementById('bone_marrow_blasts').value),
            family_history: parseInt(document.getElementById('family_history').value),
            smoking_status: parseInt(document.getElementById('smoking_status').value),
            radiation_exposure: parseInt(document.getElementById('radiation_exposure').value),
            bmi: parseFloat(document.getElementById('bmi').value),
            infection_history: parseInt(document.getElementById('infection_history').value)
        };

        try {
            // Show loading state
            resultContent.innerHTML = `
                <div style="text-align: center; padding: 2rem;">
                    <i class="fas fa-spinner fa-spin" style="font-size: 2rem; color: #d32f2f;"></i>
                    <p style="margin-top: 1rem; color: #666;">Analyzing blood sample data...</p>
                </div>
            `;
            resultContainer.style.display = 'block';

            // Send data to server
            const response = await fetch('/predict_blood_sample', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (response.ok) {
                // Display prediction result
                displayResult(result);
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        } catch (error) {
            resultContent.innerHTML = `
                <div style="text-align: center; padding: 2rem;">
                    <i class="fas fa-exclamation-circle" style="font-size: 2rem; color: #d32f2f;"></i>
                    <p style="margin-top: 1rem; color: #d32f2f; font-weight: 600;">Error: ${error.message}</p>
                    <p style="margin-top: 0.5rem; color: #666;">Please try again or contact support.</p>
                </div>
            `;
        }
    });

    function displayResult(result) {
        const prediction = result.prediction;
        const probability = result.probability ? (result.probability * 100).toFixed(2) : null;
        const isLeukemia = result.raw_prediction === 1;

        let resultHTML = `
            <div style="text-align: center;">
                <div style="display: inline-block; padding: 1rem 2rem; background: white; border-radius: 12px; border: 2px solid ${isLeukemia ? '#d32f2f' : '#4caf50'}; margin-bottom: 1rem;">
                    <p style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Diagnosis Prediction</p>
                    <p style="font-size: 1.5rem; font-weight: 700; color: ${isLeukemia ? '#d32f2f' : '#4caf50'}; margin: 0;">${prediction}</p>
                </div>
        `;

        if (probability) {
            resultHTML += `
                <div style="margin-top: 1rem;">
                    <p style="color: #666;">Confidence Level</p>
                    <div style="background: #ffebee; border-radius: 20px; height: 30px; overflow: hidden; margin-top: 0.5rem;">
                        <div style="background: linear-gradient(90deg, ${isLeukemia ? '#d32f2f, #b71c1c' : '#4caf50, #388e3c'}); height: 100%; width: ${probability}%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600; font-size: 0.9rem; transition: width 1s ease;">
                            ${probability}%
                        </div>
                    </div>
                </div>
            `;
        }

        // Add doctor consultation message based on result
        if (isLeukemia) {
            resultHTML += `
                </div>
                <div style="margin-top: 2rem; padding: 1.5rem; background: #ffebee; border-radius: 12px; border-left: 4px solid #d32f2f;">
                    <div style="display: flex; align-items: start; gap: 1rem;">
                        <i class="fas fa-exclamation-triangle" style="color: #d32f2f; font-size: 2rem; margin-top: 0.25rem;"></i>
                        <div style="text-align: left;">
                            <h3 style="color: #d32f2f; margin: 0 0 0.75rem 0; font-size: 1.2rem;"><i class="fas fa-hospital"></i> Important: Immediate Medical Attention Required</h3>
                            <p style="color: #333; font-size: 1rem; line-height: 1.8; margin: 0 0 0.75rem 0;">
                                Based on the analysis, indicators suggest possible leukemia. <strong style="color: #d32f2f;">Please consult with a healthcare professional or oncologist as soon as possible.</strong>
                            </p>
                            <p style="color: #666; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                                Early detection and prompt medical intervention are crucial for better treatment outcomes. Schedule an appointment with your doctor immediately for proper diagnosis and treatment planning.
                            </p>
                        </div>
                    </div>
                </div>
            `;
        } else {
            resultHTML += `
                </div>
                <div style="margin-top: 2rem; padding: 1.5rem; background: #e8f5e9; border-radius: 12px; border-left: 4px solid #4caf50;">
                    <div style="display: flex; align-items: start; gap: 1rem;">
                        <i class="fas fa-check-circle" style="color: #4caf50; font-size: 2rem; margin-top: 0.25rem;"></i>
                        <div style="text-align: left;">
                            <h3 style="color: #2e7d32; margin: 0 0 0.75rem 0; font-size: 1.2rem;"><i class="fas fa-heart"></i> Good News: No Leukemia Detected</h3>
                            <p style="color: #333; font-size: 1rem; line-height: 1.8; margin: 0 0 0.75rem 0;">
                                The analysis shows no indicators of leukemia. However, <strong style="color: #2e7d32;">it is still recommended to consult with your doctor for a comprehensive health checkup.</strong>
                            </p>
                            <p style="color: #666; font-size: 0.95rem; line-height: 1.6; margin: 0;">
                                Regular medical checkups are important for maintaining good health and early detection of any potential health issues. Please schedule a routine visit with your healthcare provider.
                            </p>
                        </div>
                    </div>
                </div>
            `;
        }

        resultHTML += `
            <div style="margin-top: 1.5rem; padding: 1rem; background: white; border-radius: 12px; border: 2px solid #ffcdd2;">
                <p style="color: #666; font-size: 0.9rem; line-height: 1.6; margin: 0;">
                    <strong style="color: #d32f2f;">Disclaimer:</strong> This prediction is based on machine learning analysis and should not replace professional medical diagnosis. Always consult with qualified healthcare professionals for accurate diagnosis and treatment.
                </p>
            </div>
        `;

        resultContent.innerHTML = resultHTML;
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Reset button handler
    form.addEventListener('reset', function() {
        resultContainer.style.display = 'none';
        resultContent.innerHTML = '';
    });
});
