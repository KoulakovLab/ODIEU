#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer


# DESCRIPTIONS
descriptions = [
                "This molecule smells earthy",
                "Smell reminiscent of freshly-cut grass",
               ]

# MODEL LOADING
sbert_model = "model_path"  # One of models/all-MiniLM-L12-v1/sbert_<1,2,...,5> or models/stella_en_400M_v5/sbert_<1,2,...,5>
sbert = SentenceTransformer(sbert_model, trust_remote_code=True,
                            config_kwargs={'use_memory_efficient_attention': True, 'unpad_inputs': True}
                           )

# EMBEDDINGS
embeddings = sbert.encode(descriptions)
