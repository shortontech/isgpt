package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

type InferenceRequest struct {
	Sentence string `json:"sentence"`
	Verbose  bool   `json:"verbose"`
}


func main() {
	serverURL := flag.String("server", "http://localhost:9081", "isgpt server URL")
	verbose := flag.Bool("verbose", false, "Show verbose JSON output with metrics")
	flag.Parse()

	// Require filename as positional argument
	if flag.NArg() != 1 {
		fmt.Fprintf(os.Stderr, "Usage: %s [options] <filename>\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nOptions:\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	filename := flag.Arg(0)

	// Read file
	data, err := os.ReadFile(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}
	text := string(data)

	if strings.TrimSpace(text) == "" {
		fmt.Fprintf(os.Stderr, "Error: File is empty\n")
		os.Exit(1)
	}

	// Make request to server
	result, err := analyze(text, *serverURL, *verbose)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Display results (server returns plain text by default)
	fmt.Print(result)
}

func analyze(text, serverURL string, verbose bool) (string, error) {
	reqBody := InferenceRequest{Sentence: text, Verbose: verbose}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := http.Post(serverURL+"/infer", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to connect to server: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("server returned error %d: %s", resp.StatusCode, string(body))
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	return string(body), nil
}

