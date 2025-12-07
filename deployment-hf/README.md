# CLIP Captioning - HuggingFace Deployment

Complete Docker deployment for CLIP Image Captioning using HuggingFace model.

## Model

- **HuggingFace Model**: `Hamza66628/clip-prefix-caption-coco`
- **Model Type**: CLIP Prefix Captioning (MLP mapping)
- **CLIP Model**: ViT-B/32

## Quick Start

### 1. Build and Start Services

```powershell
# Navigate to deployment directory
cd deployment-hf

# Build all services
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Access the Application

- **Frontend UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. View Logs

```powershell
# API logs
docker-compose logs -f inference-api

# Frontend logs
docker-compose logs -f frontend
```

## Services

### Inference API
- **Port**: 8000
- **Model**: Loads from HuggingFace Hub
- **GPU**: Required (NVIDIA)

### Frontend
- **Port**: 3000
- **Framework**: Vanilla HTML/CSS/JS
- **Server**: Nginx

## Configuration

### Change HuggingFace Model

Edit `docker-compose.yml`:

```yaml
environment:
  - HF_MODEL_ID=your-username/your-model-name
```

### Environment Variables

- `HF_MODEL_ID`: HuggingFace model repository ID (default: `Hamza66628/clip-prefix-caption-coco`)
- `NVIDIA_VISIBLE_DEVICES`: GPU device selection (default: `all`)

## File Structure

```
deployment-hf/
├── inference_api_hf.py      # API with HuggingFace model loading
├── Dockerfile               # API container
├── requirements-hf.txt      # Python dependencies
├── docker-compose.yml      # Service orchestration
├── frontend/
│   ├── index.html          # Frontend UI
│   ├── styles.css          # Styling
│   ├── script.js           # Frontend logic
│   ├── Dockerfile          # Frontend container
│   └── nginx.conf          # Nginx configuration
└── README.md               # This file
```

## Management Commands

```powershell
# Start services
docker-compose up -d

# Stop services
docker-compose stop

# Restart services
docker-compose restart

# Remove services
docker-compose down

# Rebuild after changes
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f

# Check resource usage
docker stats
```

## Troubleshooting

### Model Download Issues

If the model fails to download:
1. Check HuggingFace model ID is correct
2. Verify model file exists in repository (should be `pytorch_model.bin` or `model.pt`)
3. Check internet connection
4. View logs: `docker-compose logs inference-api`

### GPU Issues

If GPU is not detected:
1. Verify NVIDIA Docker runtime: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`
2. Check `runtime: nvidia` in docker-compose.yml
3. Restart Docker Desktop

### Frontend Can't Connect to API

1. Ensure API is running: `docker-compose ps`
2. Check API health: `curl http://localhost:8000/health`
3. Verify CORS is enabled in API (already configured)

## Model File Format

The HuggingFace repository should contain:
- `pytorch_model.bin` or `model.pt` - Model weights file

If using a different filename, update `inference_api_hf.py`:

```python
model_file = hf_hub_download(
    repo_id=HF_MODEL_ID,
    filename="your-model-file.pt",  # Change this
    cache_dir="./hf_cache"
)
```

## Production Deployment

For production:
1. Update CORS origins in `inference_api_hf.py`:
   ```python
   allow_origins=["https://yourdomain.com"]
   ```
2. Use environment variables for sensitive config
3. Enable HTTPS with reverse proxy (nginx/traefik)
4. Set resource limits in docker-compose.yml
5. Use Docker secrets for API keys if needed

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Verify model exists on HuggingFace
3. Ensure GPU is available and working
4. Check Docker and Docker Compose versions

