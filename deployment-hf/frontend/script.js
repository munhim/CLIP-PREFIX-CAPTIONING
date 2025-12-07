// Configuration
// API URL - connects to inference-api service
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
     ? 'http://localhost:8000'
     : `http://${window.location.hostname}:8000`;
// API URL - points to your Modal deployment
//const API_BASE_URL = 'https://hamzaimran66628--clip-captioning-api-fastapi.modal.run';

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const urlInput = document.getElementById('urlInput');
const urlSubmitBtn = document.getElementById('urlSubmitBtn');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const closePreview = document.getElementById('closePreview');
const generateBtn = document.getElementById('generateBtn');
const generateBtnText = document.getElementById('generateBtnText');
const generateBtnLoader = document.getElementById('generateBtnLoader');
const resultSection = document.getElementById('resultSection');
const captionText = document.getElementById('captionText');
const captionInfo = document.getElementById('captionInfo');
const copyBtn = document.getElementById('copyBtn');
const errorSection = document.getElementById('errorSection');
const errorText = document.getElementById('errorText');

let currentImageFile = null;
let currentImageUrl = null;

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);
urlSubmitBtn.addEventListener('click', handleUrlSubmit);
urlInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleUrlSubmit();
});
generateBtn.addEventListener('click', handleGenerate);
closePreview.addEventListener('click', resetPreview);
copyBtn.addEventListener('click', copyCaption);

// Drag and Drop Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        handleFile(files[0]);
    } else {
        showError('Please drop a valid image file');
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file');
        return;
    }

    currentImageFile = file;
    currentImageUrl = null;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        showPreview();
        hideError();
        hideResult();
    };
    reader.readAsDataURL(file);
}

// URL Handler
function handleUrlSubmit() {
    const url = urlInput.value.trim();
    if (!url) {
        showError('Please enter a valid image URL');
        return;
    }

    currentImageUrl = url;
    currentImageFile = null;
    
    // Validate URL
    try {
        new URL(url);
    } catch {
        showError('Please enter a valid URL');
        return;
    }

    previewImage.src = url;
    previewImage.onload = () => {
        showPreview();
        hideError();
        hideResult();
    };
    previewImage.onerror = () => {
        showError('Failed to load image from URL. Please check the URL and try again.');
    };
}

// Generate Caption
async function handleGenerate() {
    if (!currentImageFile && !currentImageUrl) {
        showError('Please upload an image or enter an image URL');
        return;
    }

    setLoading(true);
    hideError();
    hideResult();

    try {
        let response;
        
        if (currentImageFile) {
            // Upload file
            const formData = new FormData();
            formData.append('file', currentImageFile);
            formData.append('max_length', '77');
            formData.append('temperature', '1.0');
            formData.append('top_p', '0.9');

            response = await fetch(`${API_BASE_URL}/caption`, {
                method: 'POST',
                body: formData
            });
        } else {
            // Use URL
            response = await fetch(`${API_BASE_URL}/caption/url`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image_url: currentImageUrl,
                    max_length: 77,
                    temperature: 1.0,
                    top_p: 0.9
                })
            });
        }

        if (!response.ok) {
            let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            try {
                const errorData = await response.json();
                errorMessage = errorData.detail || errorData.message || errorMessage;
            } catch (e) {
                // If response is not JSON, use status text
                const text = await response.text().catch(() => '');
                if (text) errorMessage = text;
            }
            throw new Error(errorMessage);
        }

        const data = await response.json();
        showResult(data.caption, data.parameters || {});
    } catch (error) {
        console.error('Error generating caption:', error);
        showError(error.message || 'Failed to generate caption. Please try again.');
    } finally {
        setLoading(false);
    }
}

// UI Helpers
function showPreview() {
    previewSection.style.display = 'block';
    previewSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function resetPreview() {
    previewSection.style.display = 'none';
    fileInput.value = '';
    urlInput.value = '';
    currentImageFile = null;
    currentImageUrl = null;
    hideResult();
    hideError();
}

function showResult(caption, params) {
    captionText.textContent = caption;
    
    const infoParts = [];
    if (params.max_length) infoParts.push(`Max Length: ${params.max_length}`);
    if (params.temperature) infoParts.push(`Temperature: ${params.temperature}`);
    if (params.top_p) infoParts.push(`Top-p: ${params.top_p}`);
    
    captionInfo.textContent = infoParts.length > 0 ? `Parameters: ${infoParts.join(' | ')}` : '';
    
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResult() {
    resultSection.style.display = 'none';
}

function showError(message) {
    errorText.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    errorSection.style.display = 'none';
}

function setLoading(loading) {
    generateBtn.disabled = loading;
    generateBtnText.style.display = loading ? 'none' : 'inline';
    generateBtnLoader.style.display = loading ? 'inline-block' : 'none';
}

async function copyCaption() {
    const text = captionText.textContent;
    try {
        await navigator.clipboard.writeText(text);
        
        // Visual feedback
        const originalHTML = copyBtn.innerHTML;
        copyBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polyline points="20 6 9 17 4 12"></polyline></svg>';
        copyBtn.style.background = 'var(--success-color)';
        copyBtn.style.borderColor = 'var(--success-color)';
        
        setTimeout(() => {
            copyBtn.innerHTML = originalHTML;
            copyBtn.style.background = '';
            copyBtn.style.borderColor = '';
        }, 2000);
    } catch (err) {
        console.error('Failed to copy:', err);
        showError('Failed to copy caption to clipboard');
    }
}

// Check API health on load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('API Status:', data);
        } else {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.warn('API not reachable:', error.message);
    }
}

// Initialize
checkAPIHealth();

