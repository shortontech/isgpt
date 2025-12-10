#!/usr/bin/env python3
"""
Export GPT2 model to ONNX format for perplexity-based inference.
Disables KV cache since we need to process full sequences, not autoregressive generation.
"""
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import os

class GPT2Wrapper(torch.nn.Module):
    """Wrapper to ensure GPT-2 doesn't use past_key_values"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids):
        # Call model without past_key_values - always returns logits
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,  # Disable KV cache
            return_dict=True
        )
        return outputs.logits

def export_gpt2_to_onnx():
    model_id = "gpt2"
    output_dir = "models"

    # Create models directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading GPT2 model: {model_id}")

    # Load model and tokenizer
    base_model = GPT2LMHeadModel.from_pretrained(model_id)
    base_model.eval()  # Set to evaluation mode

    # Wrap model to disable KV cache
    model = GPT2Wrapper(base_model)
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    # Create dummy input for export
    batch_size = 1
    seq_length = 10
    dummy_input_ids = torch.randint(0, 50257, (batch_size, seq_length))
    dummy_position_ids = torch.arange(0, seq_length).unsqueeze(0)

    print("Exporting to ONNX...")

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_position_ids),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'position_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'position_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            verbose=True
        )

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)

    # Check the saved file
    if os.path.exists(onnx_path):
        size_mb = os.path.getsize(onnx_path) / 1024 / 1024
        print(f"\n[SUCCESS] Model exported successfully!")
        print(f"  ONNX model: {onnx_path} ({size_mb:.2f} MB)")
        print(f"  Tokenizer: {output_dir}/")
        print(f"\nModel signature:")
        print(f"  Inputs: input_ids [batch, seq_len], position_ids [batch, seq_len]")
        print(f"  Output: logits [batch, seq_len, vocab_size=50257]")
        print(f"  No KV cache (use_cache=False)")
    else:
        print(f"\n[ERROR] Export failed - file not found: {onnx_path}")

if __name__ == "__main__":
    export_gpt2_to_onnx()
