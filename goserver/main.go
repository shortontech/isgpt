package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"

	"github.com/daulet/tokenizers"
	ort "github.com/yalue/onnxruntime_go"
)

type GPT2Model struct {
	session   *ort.DynamicAdvancedSession
	tokenizer *tokenizers.Tokenizer
	maxLength int
	stride    int
	mu        sync.Mutex
}

type InferenceRequest struct {
	Sentence string `json:"sentence"`
	Detailed bool   `json:"detailed"`
	Verbose  bool   `json:"verbose"`
}

type SentenceDetail struct {
	Text           string  `json:"text"`
	Perplexity     float64 `json:"perplexity,omitempty"`
	Label          int     `json:"label"`
	Classification string  `json:"classification"`
	Confidence     float64 `json:"confidence"`
}

type InferenceResponse struct {
	Status            string           `json:"status,omitempty"`
	Perplexity        *float64         `json:"Perplexity,omitempty"`
	PerplexityPerLine *float64         `json:"Perplexity_per_line,omitempty"`
	Burstiness        *float64         `json:"Burstiness,omitempty"`
	Label             *int             `json:"label,omitempty"`
	Message           string           `json:"message,omitempty"`
	Sentences         []SentenceDetail `json:"sentences,omitempty"`
	MarkedText        string           `json:"marked_text,omitempty"`
}

var model *GPT2Model

func NewGPT2Model(modelPath, tokenizerPath string) (*GPT2Model, error) {
	// Initialize ONNX Runtime
	ort.SetSharedLibraryPath("/usr/lib/libonnxruntime.so")
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX runtime: %w", err)
	}

	// Load ONNX model
	inputNames := []string{"input_ids", "position_ids"}
	outputNames := []string{"logits"}

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	// Load tokenizer
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer: %w", err)
	}

	return &GPT2Model{
		session:   session,
		tokenizer: tk,
		maxLength: 1024, // GPT2's n_positions
		stride:    512,
	}, nil
}

func (m *GPT2Model) Close() {
	if m.tokenizer != nil {
		m.tokenizer.Close()
	}
	if m.session != nil {
		m.session.Destroy()
	}
	ort.DestroyEnvironment()
}

// Calculate perplexity for a given text
func (m *GPT2Model) getPPL(text string) (float64, error) {
	// Tokenize the input - Encode returns (ids []uint32, tokens []string)
	ids, _ := m.tokenizer.Encode(text, false)
	seqLen := len(ids)

	if seqLen == 0 {
		return 0, fmt.Errorf("tokenization returned empty IDs")
	}

	var nlls []float64
	prevEndLoc := 0

	for beginLoc := 0; beginLoc < seqLen; beginLoc += m.stride {
		endLoc := beginLoc + m.maxLength
		if endLoc > seqLen {
			endLoc = seqLen
		}

		trgLen := endLoc - prevEndLoc
		inputIds := ids[beginLoc:endLoc]

		// Convert to int64 for ONNX input
		inputShape := ort.NewShape(1, int64(len(inputIds)))
		tensorData := make([]int64, len(inputIds))
		for i, id := range inputIds {
			tensorData[i] = int64(id)
		}

		inputTensor, err := ort.NewTensor(inputShape, tensorData)
		if err != nil {
			return 0, fmt.Errorf("failed to create input tensor: %w", err)
		}
		defer inputTensor.Destroy()

		// Create position_ids tensor (sequential indices)
		positionData := make([]int64, len(inputIds))
		for i := range positionData {
			positionData[i] = int64(i)
		}
		positionTensor, err := ort.NewTensor(inputShape, positionData)
		if err != nil {
			return 0, fmt.Errorf("failed to create position tensor: %w", err)
		}
		defer positionTensor.Destroy()

		// Prepare output tensor
		// GPT2 output shape: [batch_size, sequence_length, vocab_size]
		vocabSize := 50257 // GPT2 vocab size
		outputShape := ort.NewShape(1, int64(len(inputIds)), int64(vocabSize))
		outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
		if err != nil {
			return 0, fmt.Errorf("failed to create output tensor: %w", err)
		}
		defer outputTensor.Destroy()

		// Run inference with both input_ids and position_ids
		err = m.session.Run([]ort.Value{inputTensor, positionTensor}, []ort.Value{outputTensor})
		if err != nil {
			return 0, fmt.Errorf("inference failed: %w", err)
		}

		// Get logits output
		logits := outputTensor.GetData()

		// Calculate negative log likelihood
		// Target is to predict next token: logits[i] predicts inputIds[i+1]
		// So for a sequence of length N, we have N-1 predictions
		// Skip the last trgLen positions for the first window (no ground truth yet)
		startIdx := 0
		if beginLoc == 0 {
			startIdx = 0
		} else {
			startIdx = len(inputIds) - trgLen
		}

		// Target IDs are the next tokens to predict
		targetIds := make([]uint32, len(inputIds)-1-startIdx)
		for i := range targetIds {
			targetIds[i] = inputIds[startIdx+i+1]
		}

		nll := m.calculateNLL(logits, targetIds, vocabSize, startIdx, len(targetIds))
		nlls = append(nlls, nll)

		prevEndLoc = endLoc
		if endLoc == seqLen {
			break
		}
	}

	// Calculate perplexity
	totalNLL := 0.0
	totalTokens := 0
	for _, nll := range nlls {
		totalNLL += nll
	}

	// Total tokens is sequence length minus 1 (we predict N-1 tokens for N input tokens)
	totalTokens = seqLen - 1
	if totalTokens <= 0 {
		totalTokens = 1
	}

	ppl := math.Exp(totalNLL / float64(totalTokens))
	return ppl, nil
}

func (m *GPT2Model) calculateNLL(logits []float32, targetIds []uint32, vocabSize int, startIdx int, count int) float64 {
	nll := 0.0

	for i := 0; i < count; i++ {
		// Get logits for position startIdx+i (predicting token at startIdx+i+1)
		offset := (startIdx + i) * vocabSize
		posLogits := logits[offset : offset+vocabSize]

		// Apply softmax and get cross-entropy loss
		targetId := int(targetIds[i])
		probs := softmax(posLogits)
		prob := float64(probs[targetId])

		// Avoid log(0)
		if prob < 1e-10 {
			prob = 1e-10
		}
		nll += -math.Log(prob)
	}

	return nll
}

func softmax(logits []float32) []float32 {
	maxLogit := logits[0]
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}

	expSum := float32(0.0)
	result := make([]float32, len(logits))

	for i, v := range logits {
		result[i] = float32(math.Exp(float64(v - maxLogit)))
		expSum += result[i]
	}

	for i := range result {
		result[i] /= expSum
	}

	return result
}

func getResults(threshold float64) (string, int, float64) {
	var label int
	var message string
	var confidence float64

	if threshold < 60 {
		label = 0
		message = "The Text is generated by AI."
		// Lower perplexity = higher AI confidence
		confidence = math.Min(100.0, (60.0-threshold)/60.0*100.0)
		if confidence < 50 {
			confidence = 50 // Minimum 50% confidence for AI classification
		}
	} else if threshold < 80 {
		label = 0
		message = "The Text is most probably contain parts which are generated by AI."
		confidence = 50.0 // Uncertain range
	} else {
		label = 1
		message = "The Text is written by Human."
		// Higher perplexity = higher human confidence
		confidence = math.Min(100.0, (threshold-80.0)/80.0*100.0)
		if confidence < 50 {
			confidence = 50 // Minimum 50% confidence for human classification
		}
	}

	return message, label, confidence
}

func (m *GPT2Model) Infer(sentence string, detailed bool) (*InferenceResponse, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	response := &InferenceResponse{}

	// Check minimum text length
	alphanumRe := regexp.MustCompile(`[a-zA-Z0-9]+`)
	matches := alphanumRe.FindAllString(sentence, -1)
	totalValidChars := 0
	for _, match := range matches {
		totalValidChars += len(match)
	}

	if totalValidChars < 100 {
		response.Status = "Please input more text (min 100 characters)"
		response.Message = "Please input more text (min 100 characters)"
		return response, nil
	}

	// Calculate overall perplexity
	ppl, err := m.getPPL(sentence)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate perplexity: %w", err)
	}
	response.Perplexity = &ppl

	// Split into sentences
	sentenceRe := regexp.MustCompile(`(?:[.?!]\s+[\[\(]?)|(?:\n\s*)`)
	lines := sentenceRe.Split(sentence, -1)

	var validLines []string
	for _, line := range lines {
		if alphanumRe.MatchString(line) {
			validLines = append(validLines, strings.TrimSpace(line))
		}
	}

	// Calculate per-line perplexity
	var perplexityPerLine []float64
	var sentenceDetails []SentenceDetail

	for _, line := range validLines {
		if line == "" {
			continue
		}

		linePPL, err := m.getPPL(line)
		if err != nil {
			log.Printf("Warning: failed to calculate PPL for line: %v", err)
			continue
		}

		perplexityPerLine = append(perplexityPerLine, linePPL)

		if detailed {
			message, label, confidence := getResults(linePPL)
			sentenceDetails = append(sentenceDetails, SentenceDetail{
				Text:           line,
				Perplexity:     linePPL,
				Label:          label,
				Classification: message,
				Confidence:     confidence,
			})
		}
	}

	if len(perplexityPerLine) == 0 {
		response.Status = "No valid sentences found"
		response.Message = "No valid sentences found"
		return response, nil
	}

	// Calculate average and max perplexity
	avgPPL := 0.0
	maxPPL := perplexityPerLine[0]
	for _, ppl := range perplexityPerLine {
		avgPPL += ppl
		if ppl > maxPPL {
			maxPPL = ppl
		}
	}
	avgPPL /= float64(len(perplexityPerLine))

	response.PerplexityPerLine = &avgPPL
	response.Burstiness = &maxPPL

	// Get final classification
	message, label, _ := getResults(avgPPL)
	response.Label = &label
	response.Message = message

	// Add detailed results if requested
	if detailed && len(sentenceDetails) > 0 {
		response.Sentences = sentenceDetails

		// Create marked text
		var markedParts []string
		for _, sent := range sentenceDetails {
			tag := "Human"
			if sent.Label == 0 {
				tag = "AI"
			}
			markedParts = append(markedParts, fmt.Sprintf("<%s>%s</%s>", tag, sent.Text, tag))
		}
		response.MarkedText = strings.Join(markedParts, " ")
	}

	return response, nil
}

func rootHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"service": "isgpt API",
		"version": "1.0",
		"endpoints": map[string]string{
			"GET /infer":  "Inference with query parameter",
			"POST /infer": "Inference with JSON body",
		},
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":       "healthy",
		"model_loaded": model != nil,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func inferHandler(w http.ResponseWriter, r *http.Request) {
	// Only accept POST
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed - use POST", http.StatusMethodNotAllowed)
		return
	}

	var req InferenceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}

	// Always request detailed to get per-sentence analysis
	result, err := model.Infer(req.Sentence, true)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Return plain text by default, JSON if verbose
	if req.Verbose {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	} else {
		w.Header().Set("Content-Type", "text/plain")

		// Format: "Sentence. <Label, confidence%>"
		var output strings.Builder
		for _, sent := range result.Sentences {
			label := "AI"
			if sent.Label == 1 {
				label = "Human"
			}
			output.WriteString(fmt.Sprintf("%s <%s, %.0f%%>\n", sent.Text, label, sent.Confidence))
		}
		output.WriteString(fmt.Sprintf("\n%s\n", result.Message))

		fmt.Fprint(w, output.String())
	}
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "9081"
	}

	host := os.Getenv("HOST")
	if host == "" {
		host = "0.0.0.0"
	}

	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "/app/models/model.onnx"
	}

	tokenizerPath := os.Getenv("TOKENIZER_PATH")
	if tokenizerPath == "" {
		tokenizerPath = "/app/models/tokenizer.json"
	}

	// Initialize model
	log.Println("Loading GPT2 model...")
	var err error
	model, err = NewGPT2Model(modelPath, tokenizerPath)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	defer model.Close()
	log.Println("Model loaded successfully!")

	// Setup HTTP routes
	http.HandleFunc("/", rootHandler)
	http.HandleFunc("/health", healthHandler)
	http.HandleFunc("/infer", inferHandler)

	addr := fmt.Sprintf("%s:%s", host, port)
	log.Printf("Starting isgpt server on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
