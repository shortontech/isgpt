# isgpt Pure Go Server

This is a pure Go implementation of the isgpt AI detection server using ONNX Runtime. **No Python runtime required!**

## Features

- ✅ Pure Go HTTP server
- ✅ ONNX Runtime CPU inference (no NVIDIA/CUDA required)
- ✅ Docker containerized
- ✅ No Python dependencies at runtime

## Architecture

- **Model**: GPT2 exported to ONNX format
- **Tokenizer**: HuggingFace tokenizer loaded via `sugarme/tokenizer`
- **Inference**: ONNX Runtime Go bindings
- **Server**: Native Go HTTP server

## Quick Start with Docker

### Build the Image

```bash
docker-compose build
```

### Run the Server

```bash
docker-compose up
```

The server will be available at `http://localhost:9081`

### Test the Server

```bash
# Health check
curl http://localhost:9081/health

# Inference with GET
curl "http://localhost:9081/infer?sentence=This%20is%20a%20test%20text&detailed=false"

# Inference with POST
curl -X POST http://localhost:9081/infer \
  -H "Content-Type: application/json" \
  -d '{"sentence": "This is a test text", "detailed": false}'
```

## API Endpoints

### GET /
Service information

### GET /health
Health check endpoint

### GET /infer
Query parameter inference
- `sentence` (required): Text to analyze
- `detailed` (optional): Include per-sentence analysis

### POST /infer
JSON body inference
```json
{
  "sentence": "Text to analyze",
  "detailed": false
}
```

## Response Format

```json
{
  "Perplexity": 45.2,
  "Perplexity_per_line": 52.3,
  "Burstiness": 120.5,
  "label": 1,
  "message": "The Text is written by Human.",
  "sentences": [...],
  "marked_text": "..."
}
```

- `label`: 0 = AI-generated, 1 = Human-written
- `sentences`: Per-sentence analysis (if `detailed=true`)
- `marked_text`: HTML-like tags marking AI/Human sections (if `detailed=true`)

## Environment Variables

- `PORT`: Server port (default: 9081)
- `HOST`: Server host (default: 0.0.0.0)
- `MODEL_PATH`: Path to ONNX model (default: /app/models/model.onnx)
- `TOKENIZER_PATH`: Path to tokenizer.json (default: /app/models/tokenizer.json)

## Development

### Export Model (One-time)

```bash
docker-compose --profile setup run model-export
```

This exports the ONNX model to a Docker volume using the HuggingFace transformers image

### Build Locally

```bash
cd goserver
go build -o isgpt-server
```

### Run Locally

```bash
MODEL_PATH=../models/model.onnx \
TOKENIZER_PATH=../models/tokenizer.json \
./isgpt-server
```

## Dependencies

### Go Libraries
- `github.com/yalue/onnxruntime_go` - ONNX Runtime bindings
- `github.com/sugarme/tokenizer` - HuggingFace tokenizer port

### System Libraries
- ONNX Runtime CPU (bundled in Docker image)
- No Python, PyTorch, or CUDA required!

## Docker Image Size

The final image is optimized with multi-stage builds:
- Builder stage: Go compiler + dependencies
- Runtime stage: Debian slim + ONNX Runtime CPU + model files

## Performance

- CPU-only inference
- No GPU required
- Suitable for production deployments on standard cloud instances

## Attribution

This implementation is an independent Go-based implementation of AI text detection using perplexity analysis. While the code in this repository is copyrighted, the underlying technique is not an original invention.

The perplexity-based approach for detecting AI-generated text is a well-documented technique. For more information on perplexity calculation with language models, see:

[Perplexity of fixed-length models - Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/perplexity)
