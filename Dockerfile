# Multi-stage Dockerfile for isgpt Pure Go Server
# Stage 1: Build Go application
FROM golang:1.21-bullseye AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Download pre-built tokenizers library v0.9.0 (matches go.mod)
RUN wget -q https://github.com/daulet/tokenizers/releases/download/v0.9.0/libtokenizers.linux-amd64.tar.gz && \
    tar -xzf libtokenizers.linux-amd64.tar.gz && \
    cp libtokenizers.a /usr/local/lib/ && \
    ldconfig && \
    rm -f libtokenizers.linux-amd64.tar.gz

# Copy go mod files
COPY goserver/go.mod goserver/go.sum* ./

# Download dependencies
RUN go mod download

# Copy source code
COPY goserver/ ./

# Build the Go binary
ENV CGO_LDFLAGS="-L/usr/local/lib"
ENV CGO_CFLAGS="-I/usr/local/include"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
RUN CGO_ENABLED=1 GOOS=linux go build -o isgpt-server .

# Stage 2: Runtime image with ONNX Runtime
FROM debian:bullseye-slim

# Install ONNX Runtime CPU and runtime dependencies
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime v1.20.0 (compatible with v1.14.0 Go bindings)
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz \
    && tar -xzf onnxruntime-linux-x64-1.20.0.tgz \
    && cp onnxruntime-linux-x64-1.20.0/lib/libonnxruntime.so* /usr/lib/ \
    && ldconfig \
    && rm -rf onnxruntime-linux-x64-1.20.0 onnxruntime-linux-x64-1.20.0.tgz

WORKDIR /app

# Copy the compiled binary from builder
COPY --from=builder /build/isgpt-server /app/

# Models will be mounted as a volume at /app/models
# No model files baked into the image

# Expose the default port
EXPOSE 9081

# Set environment variables
ENV MODEL_PATH=/app/models/model.onnx
ENV TOKENIZER_PATH=/app/models/tokenizer.json
ENV PORT=9081
ENV HOST=0.0.0.0

# Run the server
CMD ["/app/isgpt-server"]
