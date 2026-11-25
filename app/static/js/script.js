let currentDatasetId = null;
let datasetType = null;
let pipelineState = {
    upload: false,
    preprocess: false,
    eda: false,
    feature: false,
    model: false
};

function showSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => section.classList.add('d-none'));
    document.getElementById(sectionId).classList.remove('d-none');
    if (sectionId === 'dashboard') updatePipeline();
}

function updatePipeline() {
    document.getElementById('step-upload').classList.toggle('completed', pipelineState.upload);
    document.getElementById('step-preprocess').classList.toggle('completed', pipelineState.preprocess);
    document.getElementById('step-eda').classList.toggle('completed', pipelineState.eda);
    document.getElementById('step-feature').classList.toggle('completed', pipelineState.feature);
    document.getElementById('step-model').classList.toggle('completed', pipelineState.model);
    showSuggestions();
}

function showSuggestions() {
    const suggestions = document.getElementById('suggestions');
    suggestions.innerHTML = '';
    if (!pipelineState.upload) {
        suggestions.innerHTML = `
            <div class="col-md-4 mb-3">
                <div class="card suggestion-card">
                    <div class="card-body text-center">
                        <h5>Step 1: Upload Dataset</h5>
                        <p>Upload your dataset (CSV, ZIP, etc.) to begin.</p>
                        <button class="btn btn-primary" onclick="showSection('upload')">Go to Upload</button>
                    </div>
                </div>
            </div>
        `;
    } else if (!pipelineState.preprocess) {
        suggestions.innerHTML = `
            <div class="col-md-4 mb-3">
                <div class="card suggestion-card">
                    <div class="card-body text-center">
                        <h5>Step 2: Preprocess Dataset</h5>
                        <p>Process your dataset for analysis (required for audio).</p>
                        <button class="btn btn-primary" onclick="openPreprocessModal()">Start Preprocessing</button>
                    </div>
                </div>
            </div>
        `;
    } else if (!pipelineState.eda) {
        suggestions.innerHTML = `
            <div class="col-md-4 mb-3">
                <div class="card suggestion-card">
                    <div class="card-body text-center">
                        <h5>Step 3: Perform EDA</h5>
                        <p>Analyze your dataset with visualizations and summaries.</p>
                        <button class="btn btn-primary" onclick="openEDAModal()">Start EDA</button>
                    </div>
                </div>
            </div>
        `;
    } else if (!pipelineState.feature && datasetType !== 'audio') {
        suggestions.innerHTML = `
            <div class="col-md-4 mb-3">
                <div class="card suggestion-card">
                    <div class="card-body text-center">
                        <h5>Step 4: Feature Engineering</h5>
                        <p>Enhance your dataset with feature transformations.</p>
                        <button class="btn btn-primary" onclick="openFeatureModal()">Start Feature Engineering</button>
                    </div>
                </div>
            </div>
        `;
    } else if (!pipelineState.model) {
        suggestions.innerHTML = `
            <div class="col-md-4 mb-3">
                <div class="card suggestion-card">
                    <div class="card-body text-center">
                        <h5>Step 5: Train Model</h5>
                        <p>Choose an algorithm and train your model.</p>
                        <button class="btn btn-primary" onclick="openTrainModal()">Start Training</button>
                    </div>
                </div>
            </div>
        `;
    }
}

async function fetchDatasets() {
    try {
        const response = await fetch('/datasets');
        if (!response.ok) throw new Error('Failed to fetch datasets');
        const datasets = await response.json();
        const datasetList = document.getElementById('dataset-list');
        datasetList.innerHTML = '';
        datasets.forEach(dataset => {
            const card = document.createElement('div');
            card.className = 'col-md-4 mb-3';
            card.innerHTML = `
                <div class="card shadow">
                    <div class="card-body">
                        <h5 class="card-title">Dataset ${dataset.id.slice(0, 8)}</h5>
                        <p class="card-text">Type: ${dataset.type}<br>Files: ${dataset.files.length}</p>
                        <button class="btn btn-primary" onclick="selectDataset('${dataset.id}', '${dataset.type}')">Analyze</button>
                    </div>
                </div>
            `;
            datasetList.appendChild(card);
        });
    } catch (error) {
        console.error('Error fetching datasets:', error);
        document.getElementById('dataset-list').innerHTML = '<p class="text-danger">Failed to load datasets</p>';
    }
}

async function fetchModels() {
    try {
        const response = await fetch('/models');
        if (!response.ok) throw new Error('Failed to fetch models');
        const models = await response.json();
        const modelList = document.getElementById('model-list');
        modelList.innerHTML = '';
        models.forEach(model => {
            const card = document.createElement('div');
            card.className = 'col-md-4 mb-3';
            card.innerHTML = `
                <div class="card shadow">
                    <div class="card-body">
                        <h5 class="card-title">Model ${model.version}</h5>
                        <p class="card-text">Accuracy: ${model.accuracy.toFixed(2)}<br>Dataset: ${model.dataset_id.slice(0, 8)}</p>
                    </div>
                </div>
            `;
            modelList.appendChild(card);
        });
    } catch (error) {
        console.error('Error fetching models:', error);
        document.getElementById('model-list').innerHTML = '<p class="text-danger">Failed to load models</p>';
    }
}

async function uploadDataset() {
    const input = document.getElementById('dataset-input');
    const status = document.getElementById('upload-status');
    if (!input.files.length) {
        status.innerHTML = '<p class="text-danger">No file selected</p>';
        return;
    }

    const formData = new FormData();
    for (const file of input.files) {
        formData.append('files', file);
    }
    status.innerHTML = '<p>Uploading...</p>';

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        status.innerHTML = `<p class="text-success">Uploaded dataset ${data.id.slice(0, 8)}</p>`;
        pipelineState.upload = true;
        datasetType = data.type;
        updatePipeline();
        fetchDatasets();
    } catch (error) {
        console.error('Upload error:', error);
        status.innerHTML = `<p class="text-danger">Upload failed: ${error.message}</p>`;
    }
}

function selectDataset(datasetId, type) {
    currentDatasetId = datasetId;
    datasetType = type;
    pipelineState.upload = true;
    pipelineState.preprocess = false;
    pipelineState.eda = false;
    pipelineState.feature = false;
    pipelineState.model = false;
    showSection('dashboard');
    updatePipeline();
}

function openPreprocessModal() {
    if (!currentDatasetId) {
        alert('Please select a dataset first.');
        return;
    }
    if (datasetType !== 'audio') {
        alert('Preprocessing is only required for audio datasets.');
        pipelineState.preprocess = true;
        updatePipeline();
        return;
    }
    document.getElementById('preprocess-results').innerHTML = '';
    document.getElementById('preprocess-status').innerHTML = '';
    const modal = new bootstrap.Modal(document.getElementById('preprocessModal'));
    modal.show();
}

async function runPreprocess() {
    const status = document.getElementById('preprocess-status');
    const results = document.getElementById('preprocess-results');
    status.innerHTML = '<p>Processing...</p>';

    try {
        const response = await fetch(`/preprocess/${currentDatasetId}`, { method: 'POST' });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        status.innerHTML = '<p class="text-success">Preprocessing complete.</p>';
        results.innerHTML = `<p>${data.features}</p>`;
        pipelineState.preprocess = true;
        updatePipeline();
        bootstrap.Modal.getInstance(document.getElementById('preprocessModal')).hide();
    } catch (error) {
        console.error('Preprocessing error:', error);
        status.innerHTML = `<p class="text-danger">Preprocessing failed: ${error.message}</p>`;
    }
}

function skipPreprocess() {
    if (!currentDatasetId) return;
    pipelineState.preprocess = true;
    updatePipeline();
    bootstrap.Modal.getInstance(document.getElementById('preprocessModal')).hide();
}

function openEDAModal() {
    if (!currentDatasetId) {
        alert('Please select a dataset first.');
        return;
    }
    document.getElementById('eda-results').innerHTML = '';
    document.getElementById('eda-status').innerHTML = '';
    const spectrogramBtn = document.querySelector('#eda-options button[onclick="runEDA(\'spectrogram\')"]');
    spectrogramBtn.disabled = datasetType !== 'audio';
    const modal = new bootstrap.Modal(document.getElementById('edaModal'));
    modal.show();
}

async function runEDA(type) {
    const status = document.getElementById('eda-status');
    const results = document.getElementById('eda-results');
    status.innerHTML = '<p>Running...</p>';

    try {
        const response = await fetch(`/eda/${currentDatasetId}?type=${type}`, { method: 'POST' });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        status.innerHTML = '<p class="text-success">Analysis complete.</p>';
        results.innerHTML = data.visualizations.map(v => `
            <img src="${v}" class="img-fluid mb-2" style="max-width: 400px;" alt="${type} visualization">
        `).join('');
        pipelineState.eda = true;
        updatePipeline();
    } catch (error) {
        console.error('EDA error:', error);
        status.innerHTML = `<p class="text-danger">Analysis failed: ${error.message}</p>`;
    }
}

function skipEDA() {
    if (!currentDatasetId) return;
    pipelineState.eda = true;
    updatePipeline();
    bootstrap.Modal.getInstance(document.getElementById('edaModal')).hide();
}

function openFeatureModal() {
    if (!currentDatasetId) {
        alert('Please select a dataset first.');
        return;
    }
    if (datasetType === 'audio') {
        alert('Feature engineering is not supported for audio datasets.');
        pipelineState.feature = true;
        updatePipeline();
        return;
    }
    document.getElementById('feature-results').innerHTML = '';
    document.getElementById('feature-status').innerHTML = '';
    const modal = new bootstrap.Modal(document.getElementById('featureModal'));
    modal.show();
}

async function runFeatureEng(type) {
    const status = document.getElementById('feature-status');
    const results = document.getElementById('feature-results');
    status.innerHTML = '<p>Processing...</p>';

    try {
        const response = await fetch(`/feature/${currentDatasetId}?type=${type}`, { method: 'POST' });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        status.innerHTML = '<p class="text-success">Feature engineering complete.</p>';
        results.innerHTML = `<p>Processed features: ${data.features}</p>`;
        pipelineState.feature = true;
        updatePipeline();
        bootstrap.Modal.getInstance(document.getElementById('featureModal')).hide();
    } catch (error) {
        console.error('Feature engineering error:', error);
        status.innerHTML = `<p class="text-danger">Processing failed: ${error.message}</p>`;
    }
}

function skipFeatureEng() {
    if (!currentDatasetId) return;
    pipelineState.feature = true;
    updatePipeline();
    bootstrap.Modal.getInstance(document.getElementById('featureModal')).hide();
}

function openTrainModal() {
    if (!currentDatasetId) {
        alert('Please select a dataset first.');
        return;
    }
    document.getElementById('train-results').innerHTML = '';
    document.getElementById('train-status').innerHTML = '';
    document.getElementById('train-btn').disabled = false;
    fetchModelOptions();
    const modal = new bootstrap.Modal(document.getElementById('trainModal'));
    modal.show();
}

async function fetchModelOptions() {
    try {
        const response = await fetch(`/models/suggestions/${currentDatasetId}`);
        if (!response.ok) throw new Error('Failed to fetch model suggestions');
        const models = await response.json();
        const options = document.getElementById('model-options');
        options.innerHTML = models.map(m => `
            <div class="form-check">
                <input class="form-check-input" type="radio" name="model" value="${m.id}" id="${m.id}">
                <label class="form-check-label" for="${m.id}">${m.name} (${m.description})</label>
            </div>
        `).join('');
    } catch (error) {
        console.error('Error fetching model options:', error);
    }
}

async function trainModel() {
    const status = document.getElementById('train-status');
    const results = document.getElementById('train-results');
    const selectedModel = document.querySelector('input[name="model"]:checked');
    if (!selectedModel) {
        status.innerHTML = '<p class="text-danger">Please select a model.</p>';
        return;
    }

    status.innerHTML = '<p>Training...</p>';
    try {
        const response = await fetch(`/train/${currentDatasetId}?model=${selectedModel.value}`, { method: 'POST' });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        if (data.accuracy === 0 && !data.cm_path) {
            status.innerHTML = '<p class="text-danger">Training failed: Insufficient data or single label.</p>';
            results.innerHTML = '<p>Please upload a dataset with multiple samples and varied labels.</p>';
            return;
        }
        status.innerHTML = `<p class="text-success">Model ${data.version} trained.</p>`;
        results.innerHTML = `
            <p>Accuracy: ${data.accuracy.toFixed(2)}</p>
            ${data.cm_path ? `<img src="/static/visualizations/${data.cm_path.split('/').pop()}" class="img-fluid mb-2" style="max-width: 400px;" alt="Confusion Matrix">` : ''}
        `;
        document.getElementById('version-name').textContent = data.version;
        pipelineState.model = true;
        updatePipeline();
        fetchModels();
        bootstrap.Modal.getInstance(document.getElementById('trainModal')).hide();
    } catch (error) {
        console.error('Training error:', error);
        status.innerHTML = `<p class="text-danger">Training failed: ${error.message}</p>`;
    }
}

async function saveVersion() {
    const comment = document.getElementById('version-comment').value;
    try {
        const response = await fetch(`/save_version/${currentDatasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ comment })
        });
        if (!response.ok) throw new Error(await response.text());
        bootstrap.Modal.getInstance(document.getElementById('saveVersionModal')).hide();
    } catch (error) {
        console.error('Save version error:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetchDatasets();
    fetchModels();
    showSection('dashboard');
});