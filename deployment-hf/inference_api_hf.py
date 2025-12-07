"""
CLIP Prefix Captioning - Inference API (HuggingFace Model)
FastAPI-based REST API for image captioning inference using HuggingFace model
"""

import torch
import torch.nn as nn
from torch.nn import functional as nnf
import clip
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import os
from typing import Optional, Tuple
import uvicorn
import requests
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from huggingface_hub import hf_hub_download

# ========== MODEL ARCHITECTURE ==========

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

class MLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention

class TransformerLayer(nn.Module):
    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)

class Transformer(nn.Module):
    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:
                x = layer(x, y)
            elif self.enc_dec:
                x = layer(x, x, mask)
            else:
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerMapper(nn.Module):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

class ClipCaptionModel(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                 clip_length, num_layers)

# ========== GENERATION FUNCTION ==========

def generate_caption(model: ClipCaptionModel, tokenizer: GPT2Tokenizer, 
                    prefix: torch.Tensor, device: torch.device, 
                    max_length: int = 77, temperature: float = 1.0,
                    top_k: int = 0, top_p: float = 0.9):
    """Generate caption from CLIP prefix embedding"""
    model.eval()
    
    with torch.no_grad():
        prefix = prefix.to(device, dtype=torch.float32)
        
        # Project prefix to GPT embedding space
        prefix_projections = model.clip_project(prefix).view(-1, model.prefix_length, model.gpt_embedding_size)
        
        # Generate tokens
        generated_tokens = []
        generated_embeddings = prefix_projections
        
        # Start with beginning of sentence token
        tokenizer.pad_token = tokenizer.eos_token
        
        for _ in range(max_length):
            # Get logits from GPT
            outputs = model.gpt(inputs_embeds=generated_embeddings)
            logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens.append(next_token.item())
            
            # Check for end token
            if next_token.item() == tokenizer.eos_token_id or next_token.item() == tokenizer.pad_token_id:
                break
            
            # Get embedding for next token
            next_embedding = model.gpt.transformer.wte(next_token)
            if next_embedding.dim() == 2:
                next_embedding = next_embedding.unsqueeze(1)
            elif next_embedding.dim() == 3:
                if next_embedding.shape[1] > 1:
                    next_embedding = next_embedding[:, -1:, :]
            generated_embeddings = torch.cat([generated_embeddings, next_embedding], dim=1)
        
        # Decode tokens to text
        caption = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return caption.strip()

# ========== FASTAPI APP ==========

app = FastAPI(
    title="CLIP Prefix Captioning API (HuggingFace)",
    description="Generate captions for images using CLIP + GPT-2 (HuggingFace Model)",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
tokenizer = None
clip_model = None
preprocess = None

# Model configuration
HF_MODEL_ID = os.getenv('HF_MODEL_ID', 'Hamza66628/clip-prefix-caption-coco')
CLIP_MODEL_TYPE = "ViT-B/32"
PREFIX_LENGTH = 10  # Match your training config
PREFIX_SIZE = 512   # ViT-B/32 embedding size
MAPPING_TYPE = MappingType.MLP  # Match your training config

@app.on_event("startup")
async def load_models():
    """Load models on startup from HuggingFace"""
    global model, tokenizer, clip_model, preprocess
    
    print(f"Loading models on {device}...")
    print(f"HuggingFace Model ID: {HF_MODEL_ID}")
    
    try:
        # Load CLIP
        print("Loading CLIP model...")
        clip_model, preprocess = clip.load(CLIP_MODEL_TYPE, device=device, jit=False)
        clip_model.eval()
        print("✓ CLIP model loaded")
        
        # Load GPT-2 tokenizer
        print("Loading GPT-2 tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        print("✓ Tokenizer loaded")
        
        # Initialize model architecture
        print("Initializing model architecture...")
        model = ClipCaptionModel(
            prefix_length=PREFIX_LENGTH,
            prefix_size=PREFIX_SIZE,
            mapping_type=MAPPING_TYPE
        )
        
        # Download and load model from HuggingFace
        print(f"Downloading model from HuggingFace: {HF_MODEL_ID}...")
        model_file = None
        
        # Try different possible filenames
        possible_filenames = [
            "pytorch_model.bin",
            "model.pt",
            "coco_prefix_final.pt",
            "model.safetensors",
            "pytorch_model.bin.index.json"  # For sharded models
        ]
        
        for filename in possible_filenames:
            try:
                model_file = hf_hub_download(
                    repo_id=HF_MODEL_ID,
                    filename=filename,
                    cache_dir="./hf_cache"
                )
                print(f"✓ Model file found: {filename}")
                break
            except Exception as e:
                continue
        
        if model_file is None:
            # Try to list files in the repo
            try:
                from huggingface_hub import list_repo_files
                files = list_repo_files(repo_id=HF_MODEL_ID)
                print(f"Available files in repository: {files}")
                # Look for any .pt or .bin file
                for file in files:
                    if file.endswith(('.pt', '.bin', '.safetensors')):
                        try:
                            model_file = hf_hub_download(
                                repo_id=HF_MODEL_ID,
                                filename=file,
                                cache_dir="./hf_cache"
                            )
                            print(f"✓ Model file found: {file}")
                            break
                        except:
                            continue
            except Exception as e:
                pass
        
        if model_file is None:
            raise Exception(f"Could not find model file in HuggingFace repository: {HF_MODEL_ID}. Please ensure the model file (.pt, .bin, or .safetensors) exists in the repository.")
        
        # Load the model weights
        print("Loading model weights...")
        if model_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(model_file, map_location=device))
        
        model = model.to(device)
        model.eval()
        print("✓ Caption model loaded and ready from HuggingFace")
            
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "CLIP Prefix Captioning API (HuggingFace)",
        "version": "1.0.0",
        "model": HF_MODEL_ID,
        "endpoints": {
            "/caption": "POST - Generate caption for uploaded image",
            "/caption/url": "POST - Generate caption from image URL",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": model is not None,
        "clip_loaded": clip_model is not None,
        "hf_model_id": HF_MODEL_ID
    }

@app.post("/caption")
async def generate_caption_from_upload(
    file: UploadFile = File(...),
    max_length: int = 77,
    temperature: float = 1.0,
    top_p: float = 0.9
):
    """Generate caption from uploaded image file"""
    
    if model is None or clip_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract CLIP features
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = clip_model.encode_image(img_tensor).cpu().float()
        
        # Generate caption
        caption = generate_caption(model, tokenizer, clip_features, device, max_length, temperature, 0, top_p)
        
        return JSONResponse({
            "caption": caption,
            "image_size": image.size,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Request model for URL endpoint
class CaptionUrlRequest(BaseModel):
    image_url: str
    max_length: int = 77
    temperature: float = 1.0
    top_p: float = 0.9

@app.post("/caption/url")
async def generate_caption_from_url(request: CaptionUrlRequest):
    """Generate caption from image URL"""
    
    if model is None or clip_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Download image
        response = requests.get(request.image_url, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        image = Image.open(BytesIO(response.content))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract CLIP features
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_features = clip_model.encode_image(img_tensor).cpu().float()
        
        # Generate caption
        caption = generate_caption(model, tokenizer, clip_features, device, 
                                  request.max_length, request.temperature, 0, request.top_p)
        
        return JSONResponse({
            "caption": caption,
            "image_url": request.image_url,
            "image_size": image.size,
            "parameters": {
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        })
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error downloading image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image from URL: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models"""
    return {
        "caption_model": {
            "loaded": model is not None,
            "hf_model_id": HF_MODEL_ID,
            "device": str(device)
        },
        "clip_model": {
            "loaded": clip_model is not None,
            "type": CLIP_MODEL_TYPE
        },
        "tokenizer": {
            "loaded": tokenizer is not None
        }
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

