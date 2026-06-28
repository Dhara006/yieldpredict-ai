let stageChartInstance = null;
let comparisonChartInstance = null;
let climateChartInstance = null;

// Load leaderboard on page load
document.addEventListener('DOMContentLoaded', loadLeaderboard);

async function loadLeaderboard() {
    try {
        const res = await fetch('/api/models/leaderboard');
        const data = await res.json();
        const tbody = document.getElementById('leaderboard-body');
        tbody.innerHTML = '';

        data.models.forEach((m, i) => {
            const tr = document.createElement('tr');
            const timeStr = m.Time_s < 1 ? (m.Time_s * 60).toFixed(0) + 'ms' : m.Time_s.toFixed(1) + 's';
            tr.innerHTML = `
                <td>${i + 1}</td>
                <td>${m.Model} ${m.Model === data.best_model ? '<span class="best-badge">BEST</span>' : ''}</td>
                <td>${m['R²'].toFixed(4)}</td>
                <td>${m.MAE.toFixed(3)}</td>
                <td>${m.RMSE.toFixed(3)}</td>
                <td>${timeStr}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        document.getElementById('leaderboard-body').innerHTML =
            '<tr><td colspan="6" class="loading-row">Failed to load leaderboard</td></tr>';
    }
}

function updateRangeValue(id, displayId) {
    const val = document.getElementById(id).value;
    document.getElementById(displayId).textContent = val + (id === 'irrigation' ? '%' : ' kg');
}

document.getElementById('irrigation').addEventListener('input', function() {
    updateRangeValue('irrigation', 'irrigation-val');
});
document.getElementById('fertilizer').addEventListener('input', function() {
    updateRangeValue('fertilizer', 'fertilizer-val');
});

async function runPrediction() {
    const data = {
        district: document.getElementById('district').value,
        crop: document.getElementById('crop').value,
        month: parseInt(document.getElementById('month').value),
        year: 2024,
        irrigation_pct: parseFloat(document.getElementById('irrigation').value),
        fertilizer_kg_per_ha: parseFloat(document.getElementById('fertilizer').value),
        pest_history: document.getElementById('pest').value,
        sowing_area_ha: 100,
        pesticide_kg_per_ha: 10
    };

    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        const result = await response.json();

        document.getElementById('loading').style.display = 'none';
        document.getElementById('results').style.display = 'block';

        document.getElementById('district-display').textContent = '📍 ' + result.district;
        document.getElementById('crop-display').textContent = '🌾 ' + result.crop;

        const modelInfo = result.model_info || {};
        const modelBadge = document.getElementById('model-badge');
        if (modelInfo.model) {
            modelBadge.textContent = '🤖 ' + modelInfo.model + ' (R²=' + modelInfo.r2 + ')';
            modelBadge.style.display = 'inline-block';
        }

        document.getElementById('yield-value').textContent = result.yield_prediction.value_t_per_ha.toFixed(2);
        document.getElementById('risk-score').textContent = result.risk_assessment.risk_score + '/10';
        document.getElementById('risk-confidence').textContent =
            'Confidence: ' + result.risk_assessment.confidence + '%';
        document.getElementById('premium-value').textContent = '₹' +
            result.risk_assessment.premium_estimate.premium_per_ha_rupees.toLocaleString();
        document.getElementById('premium-rate').textContent =
            'Rate: ' + result.risk_assessment.premium_estimate.premium_rate_pct + '%';

        const exp = result.explanation;
        document.getElementById('yield-comparison').textContent =
            exp.summary_bullets[1] || '';

        document.getElementById('farmer-narrative').textContent = exp.narrative;
        const recContainer = document.getElementById('farmer-recommendations');
        recContainer.innerHTML = '';
        exp.recommendations.forEach(r => {
            const span = document.createElement('span');
            span.textContent = r;
            recContainer.appendChild(span);
        });

        document.getElementById('weather-summary').textContent =
            result.weather_analysis.summary || 'Weather data analyzed';
        document.getElementById('soil-summary').textContent =
            result.soil_analysis.summary || 'Soil data analyzed';
        document.getElementById('history-summary').textContent =
            result.historical_analysis.summary || 'Historical data analyzed';
        document.getElementById('risk-agent-summary').textContent =
            result.risk_assessment.summary || 'Risk analysis complete';

        renderStageChart(result.stage_predictions);
        renderComparisonChart(result.yield_prediction.value_t_per_ha, result.historical_analysis.avg_yield_5yr);
        renderClimateChart(result.risk_assessment.climate_impact);

    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        alert('Error running prediction: ' + error.message);
    }
}

function renderStageChart(stagePredictions) {
    const ctx = document.getElementById('stageChart').getContext('2d');
    const labels = Object.keys(stagePredictions).map(k => k.replace('_', ' ').replace('day ', 'Day '));
    const values = Object.values(stagePredictions);

    if (stageChartInstance) stageChartInstance.destroy();

    stageChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Predicted Yield (t/ha)',
                data: values,
                borderColor: '#2ecc71',
                backgroundColor: 'rgba(46, 204, 113, 0.1)',
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#27ae60',
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Yield (t/ha)' }
                }
            }
        }
    });
}

function renderComparisonChart(predicted, avg5yr) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');

    if (comparisonChartInstance) comparisonChartInstance.destroy();

    comparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Predicted Yield', '5-Year Average'],
            datasets: [{
                label: 'Yield (t/ha)',
                data: [predicted, avg5yr],
                backgroundColor: ['#2ecc71', '#74b9ff'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Yield (t/ha)' }
                }
            }
        }
    });
}

function renderClimateChart(climateImpact) {
    const ctx = document.getElementById('climateChart').getContext('2d');
    const labels = Object.keys(climateImpact);
    const values = Object.values(climateImpact);

    const colors = labels.map(l => {
        if (l.includes('-30%')) return '#e17055';
        if (l.includes('-10%')) return '#fdcb6e';
        return '#2ecc71';
    });

    if (climateChartInstance) climateChartInstance.destroy();

    climateChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Yield under scenario (t/ha)',
                data: values,
                backgroundColor: colors,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Yield (t/ha)' }
                }
            }
        }
    });
}
