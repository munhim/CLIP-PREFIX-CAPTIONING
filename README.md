# CLIP Prefix Captioning: Image Caption Generation with Vision-Language Models

A production-ready image captioning system that leverages CLIP (Contrastive Language-Image Pre-training) for visual feature extraction and GPT-2 for natural language generation. This project implements prefix learning to bridge visual and textual representations, enabling accurate and fluent image caption generation.

## 🎯 Project Overview

This project provides a complete implementation of CLIP Prefix Captioning, including:
- **Model Training**: Jupyter notebook for training on COCO dataset
- **REST API**: FastAPI-based inference server with HuggingFace model integration
- **Web Interface**: Modern, responsive UI for real-time caption generation
- **Deployment Options**: Docker containerization and Modal.com serverless deployment

## ✨ Features

- **Dual Architecture Support**: MLP-based and Transformer-based prefix mapping networks
- **Pre-trained Models**: Leverages CLIP (ViT-B/32) and GPT-2 for strong performance
- **Multiple Input Methods**: Upload images or provide image URLs
- **Production Ready**: Docker containerization with GPU support
- **Serverless Deployment**: Modal.com integration for scalable cloud deployment
- **Modern UI**: Responsive web interface with drag-and-drop support
- **API Documentation**: Auto-generated Swagger/OpenAPI docs

## 🏗️ Architecture

The system consists of three main components:

1. **CLIP Encoder**: Extracts visual features from input images (512-dimensional embeddings)
2. **Mapping Network**: Projects CLIP embeddings to GPT-2's embedding space (MLP or Transformer)
3. **GPT-2 Decoder**: Generates captions autoregressively conditioned on visual prefix

```
Image → CLIP Encoder → Mapping Network → Prefix Embeddings → GPT-2 → Caption
```

### Model Variants

- **Variant 1**: MLP mapping with fine-tuned GPT-2 (Best performance)
- **Variant 2**: Transformer mapping with frozen GPT-2 (Resource-efficient)

## 📋 Requirements

### For Docker Deployment
- Docker and Docker Compose
- NVIDIA Docker runtime (for GPU support)
- NVIDIA GPU with CUDA support (recommended)
- At least 4GB GPU memory

### For Modal Deployment
- Python 3.11+
- Modal account and CLI installed
- `modal` package: `pip install modal`

### For Training
- Jupyter Notebook
- GPU-enabled environment (Kaggle, Google Colab, or local GPU)
- Python 3.8+
- See `project.ipynb` for full requirements

## 🚀 Quick Start

### Option 1: Docker Deployment (Recommended for Local/On-Premises)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd GEN-AI-PROJECT
   ```

2. **Navigate to deployment directory**
   ```bash
   cd deployment-hf
   ```

3. **Build and start services**
   ```bash
   # Build containers
   docker-compose build

   # Start services
   docker-compose up -d

   # Check status
   docker-compose ps
   ```

4. **Access the application**
   - **Frontend UI**: http://localhost:3000
   - **API**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs

5. **View logs**
   ```bash
   # API logs
   docker-compose logs -f inference-api

   # Frontend logs
   docker-compose logs -f frontend
   ```

### Option 2: Modal.com Serverless Deployment

1. **Install Modal CLI**
   ```bash
   pip install modal
   modal token new
   ```

2. **Deploy to Modal**
   ```bash
   cd deployment-hf
   modal deploy modal_deploy.py
   ```

3. **Access your deployment**
   - Modal will provide a HTTPS URL (e.g., `https://username--clip-captioning-api-fastapi.modal.run`)
   - Update frontend `script.js` to point to this URL

## 📁 Project Structure

```
GEN-AI-PROJECT/
├── project.ipynb                    # Training notebook for COCO dataset
├── deployment-hf/                   # Production deployment files
│   ├── inference_api_hf.py         # FastAPI inference server
│   ├── modal_deploy.py             # Modal.com deployment script
│   ├── Dockerfile                  # Backend container definition
│   ├── docker-compose.yml          # Service orchestration
│   ├── requirements-hf.txt         # Python dependencies
│   ├── README.md                   # Deployment-specific README
│   └── frontend/                    # Web interface
│       ├── index.html              # Main HTML file
│       ├── styles.css              # Styling
│       ├── script.js               # Frontend logic
│       ├── Dockerfile              # Frontend container
│       └── nginx.conf              # Nginx configuration
└── README.md                        # This file
```

## 🔧 Configuration

### HuggingFace Model

The default model is `Hamza66628/clip-prefix-caption-coco`. To use a different model:

**Docker:**
```yaml
# Edit docker-compose.yml
environment:
  - HF_MODEL_ID=your-username/your-model-name
```

**Modal:**
```python
# Edit modal_deploy.py
HF_MODEL_ID = 'your-username/your-model-name'
```

### API Endpoints

The API provides the following endpoints:

- `POST /caption` - Generate caption from uploaded image
  ```json
  {
    "file": <image_file>,
    "max_length": 77,
    "temperature": 1.0,
    "top_p": 0.9
  }
  ```

- `POST /caption/url` - Generate caption from image URL
  ```json
  {
    "image_url": "https://example.com/image.jpg",
    "max_length": 77,
    "temperature": 1.0,
    "top_p": 0.9
  }
  ```

- `GET /health` - Health check endpoint
- `GET /models/info` - Model information
- `GET /docs` - Interactive API documentation

### Generation Parameters

- **max_length**: Maximum caption length in tokens (default: 77)
- **temperature**: Sampling temperature for text generation (default: 1.0)
  - Lower values (0.1-0.5): More deterministic, focused outputs
  - Higher values (1.0-2.0): More diverse, creative outputs
- **top_p**: Nucleus sampling parameter (default: 0.9)
  - Controls diversity by considering only tokens with cumulative probability ≤ top_p

## 🎓 Training

### Using the Jupyter Notebook

1. **Open `project.ipynb`** in a GPU-enabled environment (Kaggle, Colab, or local)

2. **Configure settings**:
   - Dataset: Uses HuggingFace `yerevann/coco-karpathy`
   - Model variant: MLP or Transformer mapping
   - Hyperparameters: Prefix length, learning rate, batch size

3. **Run training cells**:
   - The notebook handles data loading, preprocessing, and training
   - Models are saved and can be uploaded to HuggingFace Hub

4. **Upload to HuggingFace**:
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_file(
       path_or_fileobj="model.pt",
       path_in_repo="pytorch_model.bin",
       repo_id="your-username/your-model-name"
   )
   ```

### Training Hyperparameters

Based on ablation study results:
- **Prefix Length**: 5 tokens (optimal)
- **Learning Rate**: 2e-5
- **Batch Size**: 32
- **Epochs**: 10 (with early stopping)
- **Optimizer**: AdamW with linear warmup

## 🐳 Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose stop

# Restart services
docker-compose restart

# Remove services
docker-compose down

# Rebuild after code changes
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f

# Check resource usage
docker stats

# Access API container shell
docker exec -it clip-captioning-api-hf bash
```

## 🌐 Modal.com Deployment

### Initial Setup

1. **Create Modal account**: https://modal.com
2. **Install Modal CLI**: `pip install modal`
3. **Authenticate**: `modal token new`

### Deploy

```bash
cd deployment-hf
modal deploy modal_deploy.py
```

### Update Deployment

After code changes:
```bash
modal deploy modal_deploy.py
```

### Monitor

- View logs: `modal app logs clip-captioning-api`
- Check status: Modal dashboard at https://modal.com

## 🧪 Testing the API

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Generate caption from URL
curl -X POST "http://localhost:8000/caption/url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "max_length": 77,
    "temperature": 1.0,
    "top_p": 0.9
  }'

# Generate caption from file
curl -X POST "http://localhost:8000/caption" \
  -F "file=@image.jpg" \
  -F "max_length=77" \
  -F "temperature=1.0" \
  -F "top_p=0.9"
```

### Using Python

```python
import requests

# From URL
response = requests.post(
    "http://localhost:8000/caption/url",
    json={
        "image_url": "https://example.com/image.jpg",
        "max_length": 77,
        "temperature": 1.0,
        "top_p": 0.9
    }
)
print(response.json()["caption"])

# From file
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/caption",
        files={"file": f},
        data={
            "max_length": 77,
            "temperature": 1.0,
            "top_p": 0.9
        }
    )
print(response.json()["caption"])
```

## 🐛 Troubleshooting

### Model Download Issues

**Problem**: Model fails to download from HuggingFace

**Solutions**:
1. Verify model ID is correct: `Hamza66628/clip-prefix-caption-coco`
2. Check internet connection
3. Verify model file exists in repository (`pytorch_model.bin` or `model.pt`)
4. Check logs: `docker-compose logs inference-api`

### GPU Issues

**Problem**: GPU not detected

**Solutions**:
1. Verify NVIDIA Docker runtime:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
2. Check `runtime: nvidia` in `docker-compose.yml`
3. Restart Docker Desktop
4. Verify GPU is available: `nvidia-smi`

### Frontend Can't Connect to API

**Problem**: Frontend shows connection errors

**Solutions**:
1. Ensure API is running: `docker-compose ps`
2. Check API health: `curl http://localhost:8000/health`
3. Verify CORS is enabled (already configured in code)
4. Check browser console for errors
5. Verify API URL in `frontend/script.js`

### Out of Memory Errors

**Problem**: GPU out of memory during inference

**Solutions**:
1. Reduce batch size (if batching is implemented)
2. Use smaller image resolution
3. Use CPU fallback (slower but works)
4. Upgrade GPU or use cloud GPU instance

## 📊 Performance

### Model Performance (COCO Karpathy Test Set)

- **BLEU-4**: 0.352
- **METEOR**: 0.287
- **ROUGE-L**: 0.568
- **CIDEr**: 1.142

### Inference Speed

- **GPU (T4)**: ~0.5-1 second per image
- **CPU**: ~5-10 seconds per image

### Resource Requirements

- **GPU Memory**: ~4-6GB for inference
- **System RAM**: ~8GB recommended
- **Disk Space**: ~5GB for models and dependencies

## 🔐 Security Considerations

### Production Deployment

1. **CORS Configuration**: Update allowed origins in `inference_api_hf.py`
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

2. **Rate Limiting**: Implement rate limiting for API endpoints
3. **Authentication**: Add API key authentication if needed
4. **HTTPS**: Use reverse proxy (Nginx/Traefik) for SSL/TLS
5. **Input Validation**: Validate image files and URLs
6. **Resource Limits**: Set Docker resource limits in `docker-compose.yml`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **CLIP**: OpenAI's Contrastive Language-Image Pre-training model
- **GPT-2**: OpenAI's Generative Pre-trained Transformer 2
- **COCO Dataset**: Microsoft Common Objects in Context
- **HuggingFace**: For model hosting and transformers library
- **Modal**: For serverless GPU infrastructure

## 📧 Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the deployment-specific README in `deployment-hf/README.md`
- Review API documentation at `http://localhost:8000/docs`

## 📚 References

- Radford et al. "Learning Transferable Visual Models from Natural Language Supervision" (CLIP)
- Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Mokady et al. "ClipCap: CLIP Prefix for Image Captioning"
- Lin et al. "Microsoft COCO: Common Objects in Context"

---

**Built with ❤️ using CLIP, GPT-2, FastAPI, Docker, and Modal**
