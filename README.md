# isgpt - AI Text Detection

AI-generated text detection system using perplexity analysis with GPT-2. Built with Go and ONNX Runtime for efficient CPU-based inference.

## Quick Start

```bash
# First time setup: export model to Docker volume
docker-compose --profile setup run model-export

# Build and start server
docker-compose build
docker-compose up -d
```

Server runs at `http://localhost:9081`

## Usage

```bash
# Plain text output (default)
curl -X POST http://localhost:9081/infer \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Your text here...", "verbose": false}'

# JSON output with metrics
curl -X POST http://localhost:9081/infer \
  -H "Content-Type: application/json" \
  -d '{"sentence": "Your text here...", "verbose": true}'
```

**Output format (plain text)**:
```
Sentence one. <Human, 95%>
Sentence two. <AI, 75%>

The Text is written by Human.
```

**Verbose mode**: Returns JSON with perplexity metrics and per-sentence details.

## Development

Model export (one-time):
```bash
docker-compose --profile setup run model-export
```

## Attribution

This implementation is an independent Go-based implementation of AI text detection using perplexity analysis. While the code in this repository is copyrighted, the underlying technique is not an original invention.

The perplexity-based approach for detecting AI-generated text is a well-documented technique. For more information on perplexity calculation with language models, see:

[Perplexity of fixed-length models - Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/perplexity)
