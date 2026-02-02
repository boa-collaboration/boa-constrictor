import torch
import argparse
import struct
import numpy as np
import sys
import os

# Add boa folder to path to load model definition? 
# Or just load state dict generically.
# Generic loading is safer if we don't want to depend on boa code being runnable here yet.

def write_tensor(f, t):
    # Flatten and write float32
    f.write(t.float().cpu().numpy().tobytes())

def convert(model_path, output_path):
    print(f"Loading {model_path}...")
    sd = torch.load(model_path, map_location='cpu')
    if "model_state_dict" in sd:
        sd = sd["model_state_dict"] # Handle checkpoint wrappers
    
    with open(output_path, 'wb') as f:
        # 1. Embedding: embedding.weight
        # Keys might vary. Let's look for "embedding.weight"
        print("Exporting Embedding...")
        if "embedding.weight" in sd:
            write_tensor(f, sd["embedding.weight"])
        else:
            print("ERROR: embedding.weight not found")
            return

        # 2. Blocks
        # Scan for blocks.0, blocks.1 etc.
        i = 0
        while True:
            prefix = f"blocks.{i}."
            if f"{prefix}ln1.weight" not in sd:
                break
            
            print(f"Exporting Block {i}...")
            
            # Order in C++ load_weights:
            # ln1, mamba, ln2, ff
            
            # LN1
            write_tensor(f, sd[f"{prefix}ln1.weight"])
            write_tensor(f, sd[f"{prefix}ln1.bias"])
            
            # Mamba
            # Mamba keys: mamba.in_proj.weight, etc.
            # Map standard Mamba names to our C++ Loader expectation
            # C++ MambaBlock::load_weights expects:
            # in_proj.w, in_proj.b (if bias), conv1d.w, conv1d.b, ...
            # Note: The C++ Mamba we wrote loads weights in a specific order.
            # We must match src/mamba.hpp: MambaBlock::load_weights
            
            m_pre = f"{prefix}mamba."
            # in_proj
            print(f"  Writing {m_pre}in_proj.weight {sd[f'{m_pre}in_proj.weight'].shape}")
            write_tensor(f, sd[f"{m_pre}in_proj.weight"])
            
            if f"{m_pre}in_proj.bias" in sd:
                print(f"  Writing {m_pre}in_proj.bias {sd[f'{m_pre}in_proj.bias'].shape}")
                write_tensor(f, sd[f"{m_pre}in_proj.bias"])
            else:
                print(f"  Writing {m_pre}in_proj.bias (ZEROS - not found)")
                # Write zeros
                # We can't guess d_inner easily without config access.
                # But typically bias denotes existence.
                # Actually, check shape of weight [out, in]. bias is [out].
                w_shape = sd[f"{m_pre}in_proj.weight"].shape
                # w_shape is [out, in] -> bias is [out]
                bias_sim = torch.zeros(w_shape[0], dtype=torch.float32)
                write_tensor(f, bias_sim)

            # conv1d
            print(f"  Writing {m_pre}conv1d.weight")
            write_tensor(f, sd[f"{m_pre}conv1d.weight"])
            if f"{m_pre}conv1d.bias" in sd:
                 print(f"  Writing {m_pre}conv1d.bias")
                 write_tensor(f, sd[f"{m_pre}conv1d.bias"])
                 
            # x_proj
            print(f"  Writing {m_pre}x_proj.weight {sd[f'{m_pre}x_proj.weight'].shape}")
            write_tensor(f, sd[f"{m_pre}x_proj.weight"])
            
            # dt_proj
            print(f"  Writing {m_pre}dt_proj.weight {sd[f'{m_pre}dt_proj.weight'].shape}")
            write_tensor(f, sd[f"{m_pre}dt_proj.weight"])
            print(f"  Writing {m_pre}dt_proj.bias {sd[f'{m_pre}dt_proj.bias'].shape}")
            write_tensor(f, sd[f"{m_pre}dt_proj.bias"])
            
            # A_log
            print(f"  Writing {m_pre}A_log")
            write_tensor(f, sd[f"{m_pre}A_log"])
            # D
            print(f"  Writing {m_pre}D")
            write_tensor(f, sd[f"{m_pre}D"])
            
            # out_proj
            print(f"  Writing {m_pre}out_proj.weight")
            write_tensor(f, sd[f"{m_pre}out_proj.weight"])
            if f"{m_pre}out_proj.bias" in sd:
                print(f"  Writing {m_pre}out_proj.bias")
                write_tensor(f, sd[f"{m_pre}out_proj.bias"])
            else:
                print(f"  Writing {m_pre}out_proj.bias (ZEROS - not found)")
                w_shape = sd[f"{m_pre}out_proj.weight"].shape
                bias_sim = torch.zeros(w_shape[0], dtype=torch.float32)
                write_tensor(f, bias_sim)
                
            # LN2
            write_tensor(f, sd[f"{prefix}ln2.weight"])
            write_tensor(f, sd[f"{prefix}ln2.bias"])
            
            # FF
            # ff.0 (Linear), ff.2 (Linear) in Sequential
            write_tensor(f, sd[f"{prefix}ff.0.weight"])
            write_tensor(f, sd[f"{prefix}ff.0.bias"])
            write_tensor(f, sd[f"{prefix}ff.2.weight"])
            write_tensor(f, sd[f"{prefix}ff.2.bias"])
            
            i += 1

        print(f"Exported {i} blocks.")

        # 3. Head
        # head.0 (Linear), head.2 (Linear)
        print("Exporting Head...")
        write_tensor(f, sd["head.0.weight"])
        write_tensor(f, sd["head.0.bias"])
        write_tensor(f, sd["head.2.weight"])
        write_tensor(f, sd["head.2.bias"])
        
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--output", required=True, help="Path to .bin output")
    args = parser.parse_args()
    
    convert(args.model, args.output)
