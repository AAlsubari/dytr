"""
Pretrained model loader for Dynamic Transformer.
Supports loading encoder-only models (BERT, RoBERTa, DistilBERT, ALBERT) as the shared encoder.

Author: Dr. Akram Alsubari
Email: akram.alsubari@outlook.com / akram.alsubari87@gmail.com
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import requests
from tqdm import tqdm

from dytr.core.model import DynamicTransformer
from dytr.core.config import ModelConfig


class PretrainedModelLoader:
    """
    Load pretrained encoder models from HuggingFace without transformers library.
    Supports BERT, RoBERTa, DistilBERT, ALBERT architectures.
    
    These models are loaded as the shared encoder in DynamicTransformer.
    
    Example:
        >>> from dytr import PretrainedModelLoader, ModelConfig
        >>> loader = PretrainedModelLoader()
        >>> config = ModelConfig(tokenizer_name='prajjwal1/bert-tiny', use_simple_tokenizer=False)
        >>> model = loader.load_pretrained('prajjwal1/bert-tiny', config)
    """

    # Supported encoder-only model types
    SUPPORTED_MODELS = ['bert', 'roberta', 'distilbert', 'albert']
    
    # Models that are NOT supported (decoder-only or encoder-decoder)
    UNSUPPORTED_MODELS = ['gpt2', 'gpt', 't5', 'bart', 'pegasus', 'bloom', 'llama']

    def __init__(self, cache_dir: str = "./pretrained_cache"):
        """
        Initialize the pretrained model loader.
        
        Args:
            cache_dir: Directory to cache downloaded model files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_pretrained(self, model_name: str, config: Optional[ModelConfig] = None, 
                        config_override: Optional[Dict] = None) -> DynamicTransformer:
        """
        Load a pretrained encoder model from HuggingFace.
        
        Args:
            model_name: HuggingFace model name (e.g., 'prajjwal1/bert-tiny', 'bert-base-uncased')
            config: ModelConfig instance (will be created if not provided)
            config_override: Override configuration parameters
        
        Returns:
            DynamicTransformer model with pretrained encoder weights
        
        Raises:
            ValueError: If model type is not supported (decoder-only models)
        """
        # Detect and validate model type
        model_type = self._detect_model_type(model_name)
        
        if model_type in self.UNSUPPORTED_MODELS:
            raise ValueError(
                f"\n{'='*60}\n"
                f"ERROR: Model '{model_name}' is a {model_type.upper()} model which is NOT supported.\n"
                f"{'='*60}\n"
                f"DynamicTransformer only supports encoder-only models as the shared encoder.\n\n"
                f"Supported model types: {self.SUPPORTED_MODELS}\n"
                f"Examples:\n"
                f"  - BERT: 'prajjwal1/bert-tiny', 'bert-base-uncased'\n"
                f"  - RoBERTa: 'roberta-base'\n"
                f"  - DistilBERT: 'distilbert-base-uncased'\n"
                f"  - ALBERT: 'albert-base-v2'\n\n"
                f"Unsupported model types: {self.UNSUPPORTED_MODELS}\n"
                f"{'='*60}"
            )
        
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )
        
        if model_type == 'bert':
            return self.load_bert(model_name, config, config_override)
        elif model_type == 'roberta':
            return self.load_roberta(model_name, config, config_override)
        elif model_type == 'distilbert':
            return self.load_distilbert(model_name, config, config_override)
        elif model_type == 'albert':
            return self.load_albert(model_name, config, config_override)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type from name or config."""
        model_name_lower = model_name.lower()
        
        # Check by name first
        if 'bert' in model_name_lower:
            return 'bert'
        elif 'roberta' in model_name_lower:
            return 'roberta'
        elif 'distilbert' in model_name_lower:
            return 'distilbert'
        elif 'albert' in model_name_lower:
            return 'albert'
        elif 'gpt' in model_name_lower:
            return 'gpt2'
        elif 't5' in model_name_lower:
            return 't5'
        elif 'bart' in model_name_lower:
            return 'bart'
        elif 'bloom' in model_name_lower:
            return 'bloom'
        elif 'llama' in model_name_lower:
            return 'llama'
        
        # Try to detect from config if already downloaded
        try:
            config_path = self.cache_dir / model_name / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                if 'model_type' in config_data:
                    model_type = config_data['model_type']
                    if model_type in self.SUPPORTED_MODELS:
                        return model_type
                    else:
                        return model_type
        except:
            pass
        
        raise ValueError(f"Could not detect model type for {model_name}")

    def download_model_files(self, model_name: str, model_type: str) -> Dict[str, Path]:
        """
        Download model files from HuggingFace.
        
        Args:
            model_name: HuggingFace model name
            model_type: Type of model ('bert', 'roberta', etc.)

        Returns:
            Dictionary of downloaded file paths
        """
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        files_to_download = {
            'config.json': f"https://huggingface.co/{model_name}/resolve/main/config.json",
            'pytorch_model.bin': f"https://huggingface.co/{model_name}/resolve/main/pytorch_model.bin"
        }

        # Add tokenizer files based on model type
        if model_type in ['bert', 'roberta', 'distilbert', 'albert']:
            files_to_download['vocab.txt'] = f"https://huggingface.co/{model_name}/resolve/main/vocab.txt"

        downloaded_files = {}

        for filename, url in files_to_download.items():
            filepath = model_dir / filename
            if not filepath.exists():
                print(f"Downloading {filename}...")
                try:
                    self._download_file(url, filepath)
                except:
                    while True:
                        print(f"Error on Downloading A file:\n {filename}\nplease downlaod it manualy and store it as following path:\n{filepath} ")
                        input("press ENTER when completed .....")
                        if filepath.exists():
                            break
                    
                    
            else:
                print(f"Using cached {filename}")
            downloaded_files[filename] = filepath

        return downloaded_files

    def _download_file(self, url: str, filepath: Path):
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            raise

    def load_bert(self, model_name: str, 
                  config: Optional[ModelConfig] = None,
                  config_override: Optional[Dict] = None) -> DynamicTransformer:
        """
        Load BERT model as shared encoder.
        
        Args:
            model_name: BERT model name (e.g., 'prajjwal1/bert-tiny', 'bert-base-uncased')
            config: ModelConfig instance
            config_override: Override configuration parameters

        Returns:
            DynamicTransformer model with BERT encoder weights
        """
        print(f"\n{'='*60}")
        print(f"Loading BERT model: {model_name}")
        print(f"{'='*60}")
        if config and not config.tokenizer_name:
            config.tokenizer_name=model_name
        # Download model files
        files = self.download_model_files(model_name, 'bert')

        # Load config
        with open(files['config.json'], 'r') as f:
            bert_config = json.load(f)

        print(f"BERT config:")
        print(f"  Hidden size: {bert_config.get('hidden_size', 'N/A')}")
        print(f"  Layers: {bert_config.get('num_hidden_layers', 'N/A')}")
        print(f"  Attention heads: {bert_config.get('num_attention_heads', 'N/A')}")
        print(f"  Max position embeddings: {bert_config.get('max_position_embeddings', 'N/A')}")
        print(f"  Vocabulary size: {bert_config.get('vocab_size', 'N/A')}")

        # Create or update model config
        if config is None:
            config = ModelConfig(
                embed_dim=bert_config.get('hidden_size', 768),
                num_layers=bert_config.get('num_hidden_layers', 12),
                num_heads=bert_config.get('num_attention_heads', 12),
                head_dim=bert_config.get('hidden_size', 768) // bert_config.get('num_attention_heads', 12),
                ff_mult=4,
                tokenizer_name=model_name,
                #use_simple_tokenizer=False,
                max_seq_len=bert_config.get('max_position_embeddings', 512),
                dropout=bert_config.get('hidden_dropout_prob', 0.1),
                #use_rotary_embedding=False,
                #use_task_adapters=False,
                special_tokens={}
            )
        else:
            # Update config with BERT settings while preserving user settings
            #config.tokenizer_name = model_name
            #config.use_simple_tokenizer = False
            config.special_tokens = {}
            
            #if not hasattr(config, 'embed_dim') or config.embed_dim == 256:
            config.embed_dim = bert_config.get('hidden_size', config.embed_dim )
            #if not hasattr(config, 'num_layers') or config.num_layers == 6:
            config.num_layers = bert_config.get('num_hidden_layers', config.num_layers)
            #if not hasattr(config, 'num_heads') or config.num_heads == 8:
            config.num_heads = bert_config.get('num_attention_heads', config.num_heads )
            config.head_dim = config.embed_dim // config.num_heads
            #if not hasattr(config, 'max_seq_len') or config.max_seq_len == 256:
            config.max_seq_len = bert_config.get('max_position_embeddings', config.max_seq_len)
            config.use_task_adapters=False
            config.use_rotary_embedding=False
            

        # Override config if provided
        if config_override:
            for key, value in config_override.items():
                setattr(config, key, value)
                print(f"Overriding {key}: {value}")

        # Initialize DynamicTransformer
        print("\nInitializing DynamicTransformer...")
        model = DynamicTransformer(config)
        
        print(f"Tokenizer loaded with vocab size: {len(model.tokenizer)}")

        # Load weights
        print("\nLoading model weights...")
        state_dict = torch.load(files['pytorch_model.bin'], map_location='cpu')

        # Map BERT weights to encoder
        mapped_state_dict = self._map_bert_to_encoder(state_dict, config)
        
        # Load mapped weights into encoder only
        missing_keys, unexpected_keys = model.encoder.load_state_dict(mapped_state_dict, strict=False)
        
        if missing_keys:
            print(f"  Missing keys: {missing_keys}..." if len(missing_keys) > 5 else f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"  Unexpected keys: {unexpected_keys}")

        print(f"\n✓ Successfully loaded BERT model as encoder")
        print(f"  Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Heads: {config.num_heads}")
        print(f"  Vocabulary size: {len(model.tokenizer)}")
        print("\n📝 Note: This model has no Tasks: Add tasks using model.add_task() or Train the model on different Tasks to be added")

        return model

    def _map_bert_to_encoder(self, bert_state: Dict, config: ModelConfig) -> Dict:
        """
        Map BERT weights to encoder only (no decoders).
        """
        mapped = {}

        # Map word embeddings
        if 'bert.embeddings.word_embeddings.weight' in bert_state:
            src_weight = bert_state['bert.embeddings.word_embeddings.weight']
            if src_weight.shape[1] == config.embed_dim:
                mapped['embedding.weight'] = src_weight
                print(f"  Mapped word embeddings: shape {src_weight.shape}")
            else:
                print(f"  Warning: Word embedding dimension mismatch. Expected {config.embed_dim}, got {src_weight.shape[1]}")

        # Map position embeddings (if needed)
        if 'bert.embeddings.position_embeddings.weight' in bert_state:
            # BERT uses learned position embeddings, we use rotary so skip
            pass

        # Map token type embeddings (if needed)
        if 'bert.embeddings.token_type_embeddings.weight' in bert_state:
            # Not used in our model, skip
            # future work
            pass

        # Map layer norm before encoder
        if 'bert.embeddings.LayerNorm.weight' in bert_state:
            # Not used in our encoder structure, skip
            # future work
            pass

        # Map encoder layers
        for i in range(config.num_layers):  
            prefix = f'bert.encoder.layer.{i}'
            target_prefix = f'layers.{i}'
            
            # Check if this layer exists in checkpoint
            if f'{prefix}.attention.self.query.weight' not in bert_state:
                if i == 0:
                    print(f"  Warning: Layer {i} not found in checkpoint")
                continue

            # Self-attention Q, K, V projections
            if f'{prefix}.attention.self.query.weight' in bert_state:
                mapped[f'{target_prefix}.attention.q_proj.weight'] = bert_state[f'{prefix}.attention.self.query.weight']
                mapped[f'{target_prefix}.attention.q_proj.bias'] = bert_state[f'{prefix}.attention.self.query.bias']

                mapped[f'{target_prefix}.attention.k_proj.weight'] = bert_state[f'{prefix}.attention.self.key.weight']
                mapped[f'{target_prefix}.attention.k_proj.bias'] = bert_state[f'{prefix}.attention.self.key.bias']

                mapped[f'{target_prefix}.attention.v_proj.weight'] = bert_state[f'{prefix}.attention.self.value.weight']
                mapped[f'{target_prefix}.attention.v_proj.bias'] = bert_state[f'{prefix}.attention.self.value.bias']

            # Attention output projection
            if f'{prefix}.attention.output.dense.weight' in bert_state:
                mapped[f'{target_prefix}.attention.out_proj.weight'] = bert_state[f'{prefix}.attention.output.dense.weight']
                mapped[f'{target_prefix}.attention.out_proj.bias'] = bert_state[f'{prefix}.attention.output.dense.bias']

            # Attention layer norm
            if f'{prefix}.attention.output.LayerNorm.weight' in bert_state:
                mapped[f'{target_prefix}.attention_norm.weight'] = bert_state[f'{prefix}.attention.output.LayerNorm.weight']
                mapped[f'{target_prefix}.attention_norm.bias'] = bert_state[f'{prefix}.attention.output.LayerNorm.bias']

            # Feed-forward intermediate (gate and up projections)
            if f'{prefix}.intermediate.dense.weight' in bert_state:
                mapped[f'{target_prefix}.ffn.gate_proj.weight'] = bert_state[f'{prefix}.intermediate.dense.weight']
                mapped[f'{target_prefix}.ffn.gate_proj.bias'] = bert_state[f'{prefix}.intermediate.dense.bias']
                mapped[f'{target_prefix}.ffn.up_proj.weight'] = bert_state[f'{prefix}.intermediate.dense.weight']
                mapped[f'{target_prefix}.ffn.up_proj.bias'] = bert_state[f'{prefix}.intermediate.dense.bias']

            # Feed-forward output
            if f'{prefix}.output.dense.weight' in bert_state:
                mapped[f'{target_prefix}.ffn.down_proj.weight'] = bert_state[f'{prefix}.output.dense.weight']
                mapped[f'{target_prefix}.ffn.down_proj.bias'] = bert_state[f'{prefix}.output.dense.bias']

            # Feed-forward layer norm
            if f'{prefix}.output.LayerNorm.weight' in bert_state:
                mapped[f'{target_prefix}.ffn_norm.weight'] = bert_state[f'{prefix}.output.LayerNorm.weight']
                mapped[f'{target_prefix}.ffn_norm.bias'] = bert_state[f'{prefix}.output.LayerNorm.bias']

            if i < 4:  # Print first few layers
                print(f"  Mapped layer {i}")

        # Map final layer norm
        if 'bert.encoder.LayerNorm.weight' in bert_state:
            mapped['norm.weight'] = bert_state['bert.encoder.LayerNorm.weight']
            mapped['norm.bias'] = bert_state['bert.encoder.LayerNorm.bias']
            print("  Mapped final layer norm")
        else:
            
            mapped['norm.weight'] = torch.ones(config.embed_dim)
            mapped['norm.bias'] = torch.zeros(config.embed_dim)
            print("  Initialized final layer norm with identity")
        if 'bert.pooler.dense.weight' in bert_state:
            # Some BERT variants use pooler instead
            # future work
            pass

        return mapped

    def load_roberta(self, model_name: str,
                     config: Optional[ModelConfig] = None,
                     config_override: Optional[Dict] = None) -> DynamicTransformer:
        """
        Load RoBERTa model as shared encoder.
        RoBERTa uses the same architecture as BERT with different training.
        """
        print(f"\n{'='*60}")
        print(f"Loading RoBERTa model: {model_name}")
        print(f"{'='*60}")
        if config and not config.tokenizer_name:
            config.tokenizer_name=model_name
        # Download model files
        files = self.download_model_files(model_name, 'roberta')

        # Load config
        with open(files['config.json'], 'r') as f:
            roberta_config = json.load(f)

        print(f"RoBERTa config:")
        print(f"  Hidden size: {roberta_config.get('hidden_size', 'N/A')}")
        print(f"  Layers: {roberta_config.get('num_hidden_layers', 'N/A')}")
        print(f"  Attention heads: {roberta_config.get('num_attention_heads', 'N/A')}")

        # Create or update model config
        if config is None:
            config = ModelConfig(
                embed_dim=roberta_config.get('hidden_size', 768),
                num_layers=roberta_config.get('num_hidden_layers', 12),
                num_heads=roberta_config.get('num_attention_heads', 12),
                head_dim=roberta_config.get('hidden_size', 768) // roberta_config.get('num_attention_heads', 12),
                ff_mult=4,
                tokenizer_name=model_name,
                #use_simple_tokenizer=False,
                max_seq_len=roberta_config.get('max_position_embeddings', 514),
                dropout=roberta_config.get('hidden_dropout_prob', 0.1),
                use_rotary_embedding=False,
                use_task_adapters=False,
                special_tokens={}
            )
        else:
            #config.tokenizer_name = model_name
            #config.use_simple_tokenizer = False
            config.special_tokens = {}
            #if not hasattr(config, 'embed_dim') or config.embed_dim == 256:
            config.embed_dim = roberta_config.get('hidden_size', config.embed_dim)
            #if not hasattr(config, 'num_layers') or config.num_layers == 6:
            config.num_layers = roberta_config.get('num_hidden_layers', config.num_layers)
            #if not hasattr(config, 'num_heads') or config.num_heads == 8:
            config.num_heads = roberta_config.get('num_attention_heads', config.num_heads )
            config.head_dim = config.embed_dim // config.num_heads
            config.use_task_adapters=False
            config.use_rotary_embedding=False

        # Override config if provided
        if config_override:
            for key, value in config_override.items():
                setattr(config, key, value)
                print(f"Overriding {key}: {value}")

        # Initialize DynamicTransformer
        print("\nInitializing DynamicTransformer...")
        model = DynamicTransformer(config)
        
        print(f"Tokenizer loaded with vocab size: {len(model.tokenizer)}")

        # Load weights
        print("\nLoading model weights...")
        state_dict = torch.load(files['pytorch_model.bin'], map_location='cpu')

        # RoBERTa uses similar structure to BERT
        # Need to map 'roberta.' prefix to 'bert.' prefix
        renamed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('roberta.'):
                new_key = 'bert.' + key[8:]
                renamed_state_dict[new_key] = value
            else:
                renamed_state_dict[key] = value

        mapped_state_dict = self._map_bert_to_encoder(renamed_state_dict, config)
        model.encoder.load_state_dict(mapped_state_dict, strict=False)

        print(f"\n✓ Successfully loaded RoBERTa model as encoder")
        print(f"  Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Vocabulary size: {len(model.tokenizer)}")

        return model

    def load_distilbert(self, model_name: str,
                        config: Optional[ModelConfig] = None,
                        config_override: Optional[Dict] = None) -> DynamicTransformer:
        """
        Load DistilBERT model as shared encoder.
        DistilBERT is a distilled version of BERT with fewer layers.
        """
        print(f"\n{'='*60}")
        print(f"Loading DistilBERT model: {model_name}")
        print(f"{'='*60}")

        # Download model files
        files = self.download_model_files(model_name, 'distilbert')

        # Load config
        with open(files['config.json'], 'r') as f:
            distilbert_config = json.load(f)

        print(f"DistilBERT config:")
        print(f"  Hidden size: {distilbert_config.get('dim', 'N/A')}")
        print(f"  Layers: {distilbert_config.get('n_layers', 'N/A')}")
        print(f"  Attention heads: {distilbert_config.get('n_heads', 'N/A')}")

        # Create or update model config
        if config is None:
            config = ModelConfig(
                embed_dim=distilbert_config.get('dim', 768),
                num_layers=distilbert_config.get('n_layers', 6),
                num_heads=distilbert_config.get('n_heads', 12),
                head_dim=distilbert_config.get('dim', 768) // distilbert_config.get('n_heads', 12),
                ff_mult=4,
                tokenizer_name=model_name,
                #use_simple_tokenizer=False,
                max_seq_len=distilbert_config.get('max_position_embeddings', 512),
                dropout=distilbert_config.get('dropout', 0.1),
                use_rotary_embedding=False,
                use_task_adapters=False,
                special_tokens={}
            )
        else:
            #config.tokenizer_name = model_name
            #config.use_simple_tokenizer = False
            config.special_tokens = {}
            #if not hasattr(config, 'embed_dim') or config.embed_dim == 256:
            config.embed_dim = distilbert_config.get('dim', config.embed_dim)
            #if not hasattr(config, 'num_layers') or config.num_layers == 6:
            config.num_layers = distilbert_config.get('n_layers', config.num_layers )
            #if not hasattr(config, 'num_heads') or config.num_heads == 8:
            config.num_heads = distilbert_config.get('n_heads', config.num_heads)
            config.head_dim = config.embed_dim // config.num_heads
            config.use_task_adapters=False
            config.use_rotary_embedding=False

        # Override config if provided
        if config_override:
            for key, value in config_override.items():
                setattr(config, key, value)
                print(f"Overriding {key}: {value}")

        # Initialize DynamicTransformer
        print("\nInitializing DynamicTransformer...")
        model = DynamicTransformer(config)
        
        print(f"Tokenizer loaded with vocab size: {len(model.tokenizer)}")

        # Load weights
        print("\nLoading model weights...")
        state_dict = torch.load(files['pytorch_model.bin'], map_location='cpu')

        # Map DistilBERT weights to encoder
        mapped_state_dict = self._map_distilbert_to_encoder(state_dict, config)
        
        # Load mapped weights
        model.encoder.load_state_dict(mapped_state_dict, strict=False)

        print(f"\n✓ Successfully loaded DistilBERT model as encoder")
        print(f"  Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Layers: {config.num_layers}")
        print(f"  Vocabulary size: {len(model.tokenizer)}")

        return model

    def _map_distilbert_to_encoder(self, state_dict: Dict, config: ModelConfig) -> Dict:
        """
        Map DistilBERT weights to encoder.
        DistilBERT has a simpler structure than BERT.
        """
        mapped = {}

        # Word embeddings
        if 'distilbert.embeddings.word_embeddings.weight' in state_dict:
            src_weight = state_dict['distilbert.embeddings.word_embeddings.weight']
            if src_weight.shape[1] == config.embed_dim:
                mapped['embedding.weight'] = src_weight
                print(f"  Mapped word embeddings: shape {src_weight.shape}")

        # Position embeddings (if present)
        if 'distilbert.embeddings.position_embeddings.weight' in state_dict:
            # DistilBERT uses learned position embeddings, we skip
            # future work
            pass

        # Transformer layers
        for i in range(config.num_layers):
            prefix = f'distilbert.transformer.layer.{i}'
            target_prefix = f'layers.{i}'
            
            # Check if layer exists
            if f'{prefix}.attention.q_lin.weight' not in state_dict:
                continue

            # Q, K, V projections (DistilBERT has separate linear layers)
            if f'{prefix}.attention.q_lin.weight' in state_dict:
                mapped[f'{target_prefix}.attention.q_proj.weight'] = state_dict[f'{prefix}.attention.q_lin.weight']
                mapped[f'{target_prefix}.attention.q_proj.bias'] = state_dict[f'{prefix}.attention.q_lin.bias']
                
                mapped[f'{target_prefix}.attention.k_proj.weight'] = state_dict[f'{prefix}.attention.k_lin.weight']
                mapped[f'{target_prefix}.attention.k_proj.bias'] = state_dict[f'{prefix}.attention.k_lin.bias']
                
                mapped[f'{target_prefix}.attention.v_proj.weight'] = state_dict[f'{prefix}.attention.v_lin.weight']
                mapped[f'{target_prefix}.attention.v_proj.bias'] = state_dict[f'{prefix}.attention.v_lin.bias']

            # Attention output projection
            if f'{prefix}.attention.out_lin.weight' in state_dict:
                mapped[f'{target_prefix}.attention.out_proj.weight'] = state_dict[f'{prefix}.attention.out_lin.weight']
                mapped[f'{target_prefix}.attention.out_proj.bias'] = state_dict[f'{prefix}.attention.out_lin.bias']

            # Self-attention layer norm
            if f'{prefix}.sa_layer_norm.weight' in state_dict:
                mapped[f'{target_prefix}.attention_norm.weight'] = state_dict[f'{prefix}.sa_layer_norm.weight']
                mapped[f'{target_prefix}.attention_norm.bias'] = state_dict[f'{prefix}.sa_layer_norm.bias']

            # Feed-forward first linear (gate and up share same weights)
            if f'{prefix}.ffn.lin1.weight' in state_dict:
                mapped[f'{target_prefix}.ffn.gate_proj.weight'] = state_dict[f'{prefix}.ffn.lin1.weight']
                mapped[f'{target_prefix}.ffn.gate_proj.bias'] = state_dict[f'{prefix}.ffn.lin1.bias']
                mapped[f'{target_prefix}.ffn.up_proj.weight'] = state_dict[f'{prefix}.ffn.lin1.weight']
                mapped[f'{target_prefix}.ffn.up_proj.bias'] = state_dict[f'{prefix}.ffn.lin1.bias']

            # Feed-forward second linear
            if f'{prefix}.ffn.lin2.weight' in state_dict:
                mapped[f'{target_prefix}.ffn.down_proj.weight'] = state_dict[f'{prefix}.ffn.lin2.weight']
                mapped[f'{target_prefix}.ffn.down_proj.bias'] = state_dict[f'{prefix}.ffn.lin2.bias']

            # Feed-forward layer norm
            if f'{prefix}.output_layer_norm.weight' in state_dict:
                mapped[f'{target_prefix}.ffn_norm.weight'] = state_dict[f'{prefix}.output_layer_norm.weight']
                mapped[f'{target_prefix}.ffn_norm.bias'] = state_dict[f'{prefix}.output_layer_norm.bias']

            if i < 2:
                print(f"  Mapped layer {i}")

        # Final layer norm (if exists)
        if 'distilbert.transformer.LayerNorm.weight' in state_dict:
            mapped['norm.weight'] = state_dict['distilbert.transformer.LayerNorm.weight']
            mapped['norm.bias'] = state_dict['distilbert.transformer.LayerNorm.bias']
            print(f"  Mapped final layer norm")

        return mapped

    def load_albert(self, model_name: str,
                    config: Optional[ModelConfig] = None,
                    config_override: Optional[Dict] = None) -> DynamicTransformer:
        """
        Load ALBERT model as shared encoder.
        ALBERT uses parameter sharing across layers.
        """
        print(f"\n{'='*60}")
        print(f"Loading ALBERT model: {model_name}")
        print(f"{'='*60}")

        # Download model files
        files = self.download_model_files(model_name, 'albert')

        # Load config
        with open(files['config.json'], 'r') as f:
            albert_config = json.load(f)

        print(f"ALBERT config:")
        print(f"  Hidden size: {albert_config.get('hidden_size', 'N/A')}")
        print(f"  Layers: {albert_config.get('num_hidden_layers', 'N/A')}")
        print(f"  Attention heads: {albert_config.get('num_attention_heads', 'N/A')}")
        print(f"  Embedding size: {albert_config.get('embedding_size', 'N/A')}")

        # Create or update model config
        if config is None:
            config = ModelConfig(
                embed_dim=albert_config.get('hidden_size', 768),
                num_layers=albert_config.get('num_hidden_layers', 12),
                num_heads=albert_config.get('num_attention_heads', 12),
                head_dim=albert_config.get('hidden_size', 768) // albert_config.get('num_attention_heads', 12),
                ff_mult=4,
                tokenizer_name=model_name,
                #use_simple_tokenizer=False,
                max_seq_len=albert_config.get('max_position_embeddings', 512),
                dropout=albert_config.get('hidden_dropout_prob', 0.1),
                use_rotary_embedding=False,
                use_task_adapters=False,
                special_tokens={}
            )
        else:
            #config.tokenizer_name = model_name
            #config.use_simple_tokenizer = False
            config.special_tokens = {}
            
            config.embed_dim = albert_config.get('hidden_size', config.embed_dim)
            
            config.num_layers = albert_config.get('num_hidden_layers', config.num_layers)
            
            config.num_heads = albert_config.get('num_attention_heads', config.num_heads)
            config.head_dim = config.embed_dim // config.num_heads
            config.use_task_adapters=False
            config.use_rotary_embedding=False
            

        # Override config if provided
        if config_override:
            for key, value in config_override.items():
                setattr(config, key, value)
                print(f"Overriding {key}: {value}")

        # Initialize DynamicTransformer
        print("\nInitializing DynamicTransformer...")
        model = DynamicTransformer(config)
        
        print(f"Tokenizer loaded with vocab size: {len(model.tokenizer)}")

        # Load weights
        print("\nLoading model weights...")
        state_dict = torch.load(files['pytorch_model.bin'], map_location='cpu')

        # ALBERT has a similar structure to BERT but with embedding projection
        # We'll map using BERT mapping with prefix adjustments
        renamed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('albert.'):
                new_key = 'bert.' + key[7:]
                renamed_state_dict[new_key] = value
            else:
                renamed_state_dict[key] = value

        mapped_state_dict = self._map_bert_to_encoder(renamed_state_dict, config)
        model.encoder.load_state_dict(mapped_state_dict, strict=False)

        print(f"\n✓ Successfully loaded ALBERT model as encoder")
        print(f"  Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")
        print(f"  Embed dim: {config.embed_dim}")
        print(f"  Layers: {config.num_layers} (note: ALBERT shares parameters across layers)")
        print(f"  Vocabulary size: {len(model.tokenizer)}")

        return model

    def list_available_models(self) -> Dict[str, List[str]]:
        """
        List available pretrained models for encoder loading.
        
        Returns:
            Dictionary of model types with list of recommended models
        """
        return {
            'bert': [
                'prajjwal1/bert-tiny',      # 2 layers, 128 dim (~4.4M params)
                'prajjwal1/bert-mini',      # 4 layers, 256 dim (~11M params)
                'prajjwal1/bert-small',     # 4 layers, 512 dim (~28M params)
                'prajjwal1/bert-medium',    # 8 layers, 512 dim (~41M params)
                'bert-base-uncased',        # 12 layers, 768 dim (~110M params)
                'bert-large-uncased',       # 24 layers, 1024 dim (~336M params)
                'bert-base-multilingual-cased',  # Multilingual support
                'bert-base-arabic'          # Arabic BERT
            ],
            'roberta': [
                'roberta-base',             # 12 layers, 768 dim (~125M params)
                'roberta-large',            # 24 layers, 1024 dim (~355M params)
                'xlm-roberta-base',         # Multilingual RoBERTa
                'xlm-roberta-large'         # Large multilingual
            ],
            'distilbert': [
                'distilbert-base-uncased',  # 6 layers, 768 dim (~66M params)
                'distilbert-base-multilingual-cased'  # Multilingual
            ],
            'albert': [
                'albert-base-v2',           # 12 layers, 768 dim (~12M params)
                'albert-large-v2',          # 24 layers, 1024 dim (~18M params)
                'albert-xlarge-v2',         # 24 layers, 2048 dim (~60M params)
                'albert-xxlarge-v2'         # 12 layers, 4096 dim (~235M params)
            ]
        }

    def get_model_info(self, model_name: str) -> Dict:
        """
        Get information about a pretrained model without loading it.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Dictionary with model information
        """
        try:
            model_type = self._detect_model_type(model_name)
            
            # Try to download config
            model_dir = self.cache_dir / model_name
            config_path = model_dir / 'config.json'
            
            if not config_path.exists():
                url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    config = response.json()
                else:
                    return {"error": "Could not fetch config"}
            else:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            info = {
                "model_name": model_name,
                "model_type": model_type,
                "supported": model_type in self.SUPPORTED_MODELS,
                "architecture": {}
            }
            
            if model_type == 'bert':
                info["architecture"] = {
                    "hidden_size": config.get('hidden_size', 'N/A'),
                    "num_layers": config.get('num_hidden_layers', 'N/A'),
                    "num_heads": config.get('num_attention_heads', 'N/A'),
                    "vocab_size": config.get('vocab_size', 'N/A'),
                    "max_position_embeddings": config.get('max_position_embeddings', 'N/A')
                }
            elif model_type == 'roberta':
                info["architecture"] = {
                    "hidden_size": config.get('hidden_size', 'N/A'),
                    "num_layers": config.get('num_hidden_layers', 'N/A'),
                    "num_heads": config.get('num_attention_heads', 'N/A'),
                    "vocab_size": config.get('vocab_size', 'N/A'),
                    "max_position_embeddings": config.get('max_position_embeddings', 'N/A')
                }
            elif model_type == 'distilbert':
                info["architecture"] = {
                    "hidden_size": config.get('dim', 'N/A'),
                    "num_layers": config.get('n_layers', 'N/A'),
                    "num_heads": config.get('n_heads', 'N/A'),
                    "vocab_size": config.get('vocab_size', 'N/A'),
                    "max_position_embeddings": config.get('max_position_embeddings', 'N/A')
                }
            elif model_type == 'albert':
                info["architecture"] = {
                    "hidden_size": config.get('hidden_size', 'N/A'),
                    "num_layers": config.get('num_hidden_layers', 'N/A'),
                    "num_heads": config.get('num_attention_heads', 'N/A'),
                    "vocab_size": config.get('vocab_size', 'N/A'),
                    "embedding_size": config.get('embedding_size', 'N/A')
                }
            
            return info
            
        except Exception as e:
            return {"error": str(e)}