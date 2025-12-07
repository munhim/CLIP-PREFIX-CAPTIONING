# Modal.com Deployment Guide

This guide will help you deploy your CLIP Prefix Captioning model on Modal.com with a T4 GPU.

## Prerequisites

1. Modal account with token configured (you've already done this!)
2. Modal CLI installed: `pip install modal`
3. Python 3.8+

## Deployment Steps

### 1. Navigate to the deployment directory

```powershell
cd GEN-AI-PROJECT\deployment-hf
```

### 2. Deploy to Modal

```powershell
modal deploy modal_deploy.py
```

This will:
- Build the container image with all dependencies
- Deploy the FastAPI app to Modal
- Attach a T4 GPU to the deployment
- Return a public URL for your API

### 3. Access Your Deployed API

After deployment, Modal will provide you with a public URL like:
```
https://<your-username>--clip-captioning-api-fastapi.modal.run
```

## API Endpoints

Once deployed, your API will have the following endpoints:

### Root
- `GET /` - API information

### Health Check
- `GET /health` - Check if models are loaded and API is healthy

### Generate Caption from URL
- `POST /caption/url` - Generate caption from an image URL
  ```json
  {
    "image_url": "https://example.com/image.jpg",
    "max_length": 77,
    "temperature": 1.0,
    "top_p": 0.9
  }
  ```

### Generate Caption from Upload
- `POST /caption` - Generate caption from uploaded image file
  - Content-Type: `multipart/form-data`
  - Parameters:
    - `file`: Image file
    - `max_length`: Maximum caption length (default: 77)
    - `temperature`: Generation temperature (default: 1.0)
    - `top_p`: Top-p sampling (default: 0.9)

### Model Info
- `GET /models/info` - Get information about loaded models

### API Documentation
- `GET /docs` - Interactive Swagger UI documentation

## Example Usage

### Using cURL

```bash
# Health check
curl https://<your-username>--clip-captioning-api-fastapi.modal.run/health

# Generate caption from URL
curl -X POST https://<your-username>--clip-captioning-api-fastapi.modal.run/caption/url \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'

# Generate caption from file upload
curl -X POST https://<your-username>--clip-captioning-api-fastapi.modal.run/caption \
  -F "file=@image.jpg" \
  -F "max_length=77" \
  -F "temperature=1.0" \
  -F "top_p=0.9"
```

### Using Python

```python
import requests

# Health check
response = requests.get("https://<your-username>--clip-captioning-api-fastapi.modal.run/health")
print(response.json())

# Generate caption from URL
response = requests.post(
    "https://<your-username>--clip-captioning-api-fastapi.modal.run/caption/url",
    json={
        "image_url": "https://example.com/image.jpg",
        "max_length": 77,
        "temperature": 1.0,
        "top_p": 0.9
    }
)
print(response.json())

# Generate caption from file
with open("image.jpg", "rb") as f:
    response = requests.post(
        "https://<your-username>--clip-captioning-api-fastapi.modal.run/caption",
        files={"file": f},
        data={"max_length": 77, "temperature": 1.0, "top_p": 0.9}
    )
print(response.json())
```

## Configuration

### Change HuggingFace Model

Edit `modal_deploy.py` and modify the `HF_MODEL_ID` variable:

```python
HF_MODEL_ID = os.getenv('HF_MODEL_ID', 'your-username/your-model-name')
```

### Environment Variables

You can set environment variables when deploying:

```powershell
modal deploy modal_deploy.py --env HF_MODEL_ID=your-username/your-model-name
```

## Monitoring

### View Logs

```powershell
modal app logs clip-captioning-api
```

### View App Status

```powershell
modal app list
```

### Stop/Start App

```powershell
# Stop the app (but keep warm containers running)
modal app stop clip-captioning-api

# Resume the app
modal app resume clip-captioning-api
```

## Costs

Modal charges based on:
- GPU usage time (T4 GPU)
- Container idle time (keep_warm=1 keeps one container warm)
- Storage for model cache

The `keep_warm=1` setting keeps one container warm to reduce cold start latency, but this will incur continuous costs. Remove it or set to 0 if you want to save costs (but expect ~30-60 second cold starts).

## Troubleshooting

### Model Not Loading

1. Check logs: `modal app logs clip-captioning-api`
2. Verify HuggingFace model ID is correct
3. Ensure model file exists in the HuggingFace repository

### GPU Not Available

1. Check Modal dashboard for GPU availability
2. T4 GPUs are usually available, but may have wait times during peak usage
3. You can try a different GPU type by changing `gpu.T4(count=1)` to `gpu.A10G(count=1)` in the code

### Cold Start Issues

If you experience long cold starts:
- Increase `keep_warm` value (costs more but faster)
- Reduce `container_idle_timeout` if you expect regular traffic

## Additional Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Python SDK Reference](https://modal.com/docs/reference)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

