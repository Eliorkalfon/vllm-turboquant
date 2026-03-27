#!/usr/bin/env python3
import argparse
from pathlib import Path

# Import the existing helper functions from vllm-turboquant
from benchmarks.generate_turboquant_metadata import _derive_model_shape, _resolve_layer_indices
from vllm.v1.attention.ops.turboquant_metadata import build_default_turboquant_metadata, save_turboquant_metadata

def main():
    parser = argparse.ArgumentParser(description="Generate a fallback dummy TurboQuant metadata file without loading model weights.")
    parser.add_argument("--model", required=True, help="HF model ID or local path (e.g., your-org/gpt-oss-120b)")
    parser.add_argument("--kv-cache-dtype", choices=("turboquant25", "turboquant35"), default="turboquant35")
    parser.add_argument("--output", default="./turboquant_kv.json")
    parser.add_argument("--layer-pattern", default="model.layers.{i}.self_attn.attn")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    print(f"Fetching config for {args.model} (this only downloads a tiny config.json)...")
    head_size, num_kv_heads, num_hidden_layers, layer_types, _ = _derive_model_shape(args.model, trust_remote_code=args.trust_remote_code)
    
    # Ignore strict layer type filtering. For dummy/fallback inference, generate keys for all layers.
    layer_indices = range(num_hidden_layers)
    layer_names = [args.layer_pattern.format(i=idx) for idx in layer_indices]

    print(f"Found {num_hidden_layers} layers, {num_kv_heads} KV heads, and head size {head_size}.")
    print("Building default/dummy metadata (first N channels assumed as outliers)...")

    metadata = build_default_turboquant_metadata(
        recipe=args.kv_cache_dtype,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        layer_names=layer_names,
        model_name=args.model,
    )
    
    save_turboquant_metadata(metadata, Path(args.output))
    print(f"Success! Dummy metadata generated and saved to {args.output}")

if __name__ == "__main__":
    main()
