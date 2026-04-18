#!/usr/bin/env python3
"""
CCLMoff model definition for the local benchmark 
Simplified architecture for benchmarking baseline:
    RNA-FM t12 backbone (frozen)
    -> CLS embedding (640)
    -> Linear(640, 64) -> ReLU -> Dropout(0.5)
    -> Linear(64, 1) -> Sigmoid

Key changes from original:
- Higher dropout (0.3 instead of 0.2)
- Simpler activation (ReLU instead of ELU)
- allow finetuning of the RNA-FM backbone (unfreeze parameters)
"""

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


class CCLMoff(nn.Module if nn is not None else object):
    """RNA-FM backbone with simplified MLP head - downgraded for benchmarking."""

    def __init__(self):
        if TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "CCLMoff requires PyTorch. Install `torch` before constructing the model."
            ) from TORCH_IMPORT_ERROR

        super().__init__()

        try:
            import fm
        except ImportError as exc:
            raise ImportError(
                "CCLMoff requires the official RNA-FM package. Install `rna-fm` "
                "so `fm.pretrained.rna_fm_t12()` is available."
            ) from exc

        self.rna_model, self.rna_alphabet = fm.pretrained.rna_fm_t12()
        
        for param in self.rna_model.parameters():
            param.requires_grad = True

        self.dense1 = nn.Linear(640, 64)
        self.dense2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.act = nn.Sigmoid()

        for module in self.children():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data)

    def get_alphabet(self):
        return self.rna_alphabet

    def forward(self, tokens):
        with torch.no_grad():
            rna_results = self.rna_model(tokens, repr_layers=[12])
        seq_emb = rna_results["representations"][12][:, 0, :]

        hidden = self.relu(self.dropout(self.dense1(seq_emb)))
        logits = self.dense2(self.dropout(hidden))
        return self.act(logits)
