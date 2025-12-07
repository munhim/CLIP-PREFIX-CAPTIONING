# How to Run the Frontend App

Your CLIP Captioning frontend app is ready to use with your Modal deployment!

## Option 1: Open Directly in Browser (Simplest)

1. Navigate to the frontend folder:
   ```powershell
   cd GEN-AI-PROJECT\deployment-hf\frontend
   ```

2. Open `index.html` in your web browser:
   - Double-click the file, or
   - Right-click → "Open with" → Your browser

The app will connect to your Modal API at:
`https://hamzaimran66628--clip-captioning-api-fastapi.modal.run`

## Option 2: Use a Simple HTTP Server (Recommended)

For better compatibility, use a local HTTP server:

### Using Python (if installed):
```powershell
cd GEN-AI-PROJECT\deployment-hf\frontend
python -m http.server 8080
```
Then open: http://localhost:8080

### Using Node.js (if installed):
```powershell
cd GEN-AI-PROJECT\deployment-hf\frontend
npx http-server -p 8080
```
Then open: http://localhost:8080

### Using PHP (if installed):
```powershell
cd GEN-AI-PROJECT\deployment-hf\frontend
php -S localhost:8080
```
Then open: http://localhost:8080

## Option 3: Deploy Frontend to Static Hosting

You can deploy the frontend to any static hosting service:
- **GitHub Pages**: Free hosting for static sites
- **Netlify**: Drag and drop deployment
- **Vercel**: Easy deployment
- **Cloudflare Pages**: Fast global CDN

## Using the App

1. **Upload an Image**: 
   - Drag & drop an image onto the upload area, OR
   - Click to browse and select an image

2. **Or Use Image URL**:
   - Enter an image URL (e.g., `https://example.com/image.jpg`)
   - Click "Generate from URL"

3. **Generate Caption**:
   - Click the "Generate Caption" button
   - Wait for the AI to generate a caption (first request may take 30-60 seconds)
   - The caption will appear below the image

4. **Copy Caption**:
   - Click the copy button to copy the caption to clipboard

## Troubleshooting

### CORS Errors
If you see CORS errors, the Modal API has CORS enabled, so this shouldn't happen. If it does:
- Make sure you're using the correct Modal URL
- Check browser console for specific error messages

### API Not Responding
- Check that your Modal deployment is active: https://modal.com/apps/hamzaimran66628/main/deployed/clip-captioning-api
- Test the API directly: https://hamzaimran66628--clip-captioning-api-fastapi.modal.run/health

### First Request is Slow
The first request takes 30-60 seconds because the models need to load. Subsequent requests will be much faster!

## Testing the API Directly

You can also test the API using curl or Python:

### Using curl:
```bash
# Health check
curl https://hamzaimran66628--clip-captioning-api-fastapi.modal.run/health

# Generate caption from URL
curl -X POST https://hamzaimran66628--clip-captioning-api-fastapi.modal.run/caption/url \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/image.jpg"}'
```

### Using Python:
```python
import requests

# Health check
response = requests.get('https://hamzaimran66628--clip-captioning-api-fastapi.modal.run/health')
print(response.json())

# Generate caption from URL
response = requests.post(
    'https://hamzaimran66628--clip-captioning-api-fastapi.modal.run/caption/url',
    json={
        "image_url": "https://example.com/image.jpg",
        "max_length": 77,
        "temperature": 1.0,
        "top_p": 0.9
    }
)
print(response.json())
```

