


import json
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
import requests
from tqdm import tqdm

from dytr.core.model import DynamicTransformer
from dytr.core.config import ModelConfig, TaskConfig, TrainingStrategy


class PretrainedModelLoader:

    def __init__(self, cache_dir: str = "./pretrained_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = None
        self.model_name = None
        self.vocab_path=None
        self.vocab_size=None
        self.device="cuda" if __import__("torch").cuda.is_available() else "cpu"

    def load_pretrained(
        self,
        model_name: str,
        config: Optional[ModelConfig] = None,
        config_override: Optional[Dict] = None,
        task_name: str = None,
        load_mode: str = "auto"
    ) -> Tuple[DynamicTransformer, Optional[TaskConfig]]:

        self.task_name = task_name
        self.model_name = model_name

        try:
            architecture = self._detect_architecture(model_name)
        except Exception as e:
            print(f"Warning: Could not detect architecture: {e}")
            architecture = "encoder_only"

        print(f"\n{'='*60}")
        print(f"Detected architecture: {architecture}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        try:
            if architecture == "encoder_only":
                model, _ = self._load_encoder_only(model_name, config, config_override)
                return model#, None

            elif architecture == "decoder_only":
                model, task_config = self._load_decoder_only(model_name, config, config_override)
                return model#, task_config

            elif architecture == "encoder_decoder":
                if load_mode == "encoder_only":
                    model, _ = self._load_encoder_only(model_name, config, config_override)
                    return model#, None
                else:
                    model, task_config = self._load_encoder_decoder(model_name, config, config_override)
                    return model#, task_config

            else:
                print(f"Unknown architecture: {architecture}, trying encoder_only mode")
                model, _ = self._load_encoder_only(model_name, config, config_override)
                return model#, None

        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load {model_name}: {e}")

    def _detect_architecture(self, model_name: str) -> str:
        try:
            config_path = self.cache_dir / model_name / 'config.json'
            if not config_path.exists():
                self._download_config(model_name)

            with open(config_path, 'r') as f:
                config = json.load(f)

            model_type = config.get('model_type', '').lower()
            architectures = config.get('architectures', [])

            encoder_only = ['roberta', 'distilbert', 'albert', 'deberta', 'electra', 'camembert', 'xlm-roberta', 'xlm', 'bert']
            for t in encoder_only:
                if t in model_type:
                    return "encoder_only"

            decoder_only = ['gpt2', 'gpt', 'llama', 'mistral', 'phi', 'gemma', 'bloom', 'opt', 'falcon', 'qwen']
            for t in decoder_only:
                if t in model_type:
                    return "decoder_only"

            encoder_decoder = ['longt5', 'mt5', 'flan-t5', 'flan', 't5', 'mbart', 'bart', 'pegasus', 'led', 'prophetnet']
            for t in encoder_decoder:
                if t in model_type:
                    return "encoder_decoder"

            if architectures:
                arch_str = str(architectures).lower()
                if 'forconditionalgeneration' in arch_str or 'seq2seq' in arch_str:
                    return "encoder_decoder"
                if 'lmhead' in arch_str or 'causallm' in arch_str:
                    return "decoder_only"
                if 'maskedlm' in arch_str or 'forpretraining' in arch_str:
                    return "encoder_only"

            return "encoder_only"

        except Exception as e:
            print(f"Architecture detection error: {e}")
            return "encoder_only"

    def _download_config(self, model_name: str):
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        filepath = model_dir / 'config.json'

        if not filepath.exists():
            self._download_file(url, filepath)

    def _check_file_exists(self, url: str) -> bool:
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                return True
            if response.status_code == 302 or response.status_code == 307:
                location = response.headers.get('Location', '')
                if location:
                    return self._check_file_exists(location)
            return False
        except Exception:
            try:
                response = requests.get(url, stream=True, timeout=10)
                response.close()
                return response.status_code == 200
            except Exception:
                return False
    def _find_model_files(self, model_name: str, model_dir: Path) -> List[Path]:
        model_files = []
        try:
            safetensors_files = list(model_dir.glob("*.safetensors"))
            if safetensors_files:
                model_files.extend(safetensors_files)

            bin_files = list(model_dir.glob("pytorch_model*.bin"))
            if bin_files:
                model_files.extend(bin_files)

            adapter_files = list(model_dir.glob("adapter_model*.bin"))
            if adapter_files:
                model_files.extend(adapter_files)

        except Exception as e:
            print(f"Warning: Error finding model files: {e}")

        return model_files
    def _download_model_files(self, model_name: str, model_type: str) -> Dict[str, Path]:
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_path=self._download_tokenizer_files(model_name,model_dir)
    
        downloaded_files = {}
    
        config_url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        config_path = model_dir / 'config.json'
        if not config_path.exists():
            print(f"Downloading config.json...")
            self._download_file(config_url, config_path)
        downloaded_files['config.json'] = config_path
    
        model_files = self._find_model_files(model_name, model_dir)
        if model_files:
            print(f"Found existing model files in cache")
            downloaded_files['model_files'] = model_files
            return downloaded_files
    
        index_files = ['pytorch_model.bin.index.json', 'model.safetensors.index.json']
        for index_file in index_files:
            index_url = f"https://huggingface.co/{model_name}/resolve/main/{index_file}"
            if self._check_file_exists(index_url):
                print(f"Found sharded index: {index_file}")
                index_path = model_dir / index_file
                self._download_file(index_url, index_path)
                downloaded_files['index'] = index_path
    
                with open(index_path, 'r') as f:
                    index_data = json.load(f)
    
                shard_files = set(index_data.get('weight_map', {}).values())
                for shard in shard_files:
                    shard_url = f"https://huggingface.co/{model_name}/resolve/main/{shard}"
                    shard_path = model_dir / shard
                    if not shard_path.exists():
                        print(f"Downloading shard: {shard}")
                        self._download_file(shard_url, shard_path)
                    downloaded_files[shard] = shard_path
                return downloaded_files
    
        single_candidates = [
            ('model.safetensors', 'model.safetensors'),
            ('pytorch_model.bin', 'pytorch_model.bin'),
            ('adapter_model.bin', 'adapter_model.bin')
        ]
    
        for remote_name, local_name in single_candidates:
            url = f"https://huggingface.co/{model_name}/resolve/main/{remote_name}"
            if self._check_file_exists(url):
                print(f"Downloading {remote_name}...")
                filepath = model_dir / local_name
                self._download_file(url, filepath)
                downloaded_files['model_file'] = filepath
                return downloaded_files
    
        valid_attempts = [f"'{name}'" for name, _ in single_candidates]
        raise FileNotFoundError(
            f"Could not find model file for {model_name}. "
            f"Tried: {', '.join(valid_attempts)} and sharded indices."
        )
    

    def _download_file(self, url: str, filepath: Path):
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

    def _load_state_dict_from_files(self, model_dir: Path, model_files: List[Path]) -> Dict:
        state_dict = {}

        safetensors_available = False
        try:
            from safetensors.torch import load_file as load_safetensors
            safetensors_available = True
        except ImportError:
            pass

        for file_path in model_files:
            try:
                ext = file_path.suffix.lower()

                if ext == '.safetensors':
                    if safetensors_available:
                        shard_dict = load_safetensors(file_path)
                        state_dict.update(shard_dict)
                        print(f"  Loaded {file_path.name}")
                    else:
                        print(f"  Warning: safetensors file found but safetensors package not installed")
                        print(f"  Install with: pip install safetensors")

                elif ext == '.bin':
                    shard_dict = torch.load(file_path, map_location=self.device)
                    state_dict.update(shard_dict)
                    print(f"  Loaded {file_path.name}")

            except Exception as e:
                print(f"  Warning: Failed to load {file_path.name}: {e}")

        return state_dict

    def _load_encoder_only(self, model_name: str, config: Optional[ModelConfig] = None,
                           config_override: Optional[Dict] = None) -> Tuple[DynamicTransformer, None]:

        print("Loading as ENCODER-ONLY → Shared encoder")

        try:
            self._download_config(model_name)

            with open(self.cache_dir / model_name / 'config.json', 'r') as f:
                model_config = json.load(f)

            config, _ = self._extract_config_params(model_config, config)

            if config_override:
                for key, value in config_override.items():
                    setattr(config, key, value)

            model = DynamicTransformer(config)
            config=model.config
            

            model_dir = self.cache_dir / model_name
            model_files = self._find_model_files(model_name, model_dir)

            if not model_files:
                files = self._download_model_files(model_name, 'encoder_only')
                model_files = files.get('model_files', [])
                if 'model_file' in files:
                    model_files = [files['model_file']]
                elif 'index' in files:
                    model_files = [f for f in files.values() if f.suffix in ['.bin', '.safetensors']]

            if not model_files:
                raise FileNotFoundError(f"No model files found for {model_name}")

            state_dict = self._load_state_dict_from_files(model_dir, model_files)
            mapped_state_dict = self._map_encoder_weights(state_dict, config, model)
            model.encoder.load_state_dict(mapped_state_dict, strict=False)

            print(f"✓ Loaded {model_name} as shared encoder")
            print(f"  Parameters: {sum(p.numel() for p in model.encoder.parameters()):,}")

            return model, None

        except Exception as e:
            print(f"Error loading encoder-only model: {e}")
            raise

    def _load_decoder_only(self, model_name: str, config: Optional[ModelConfig] = None,
                           config_override: Optional[Dict] = None) -> Tuple[DynamicTransformer, TaskConfig]:

        print("Loading as DECODER-ONLY → Creating Causal LM task")

        try:
            self._download_config(model_name)

            with open(self.cache_dir / model_name / 'config.json', 'r') as f:
                model_config = json.load(f)

            config, _ = self._extract_config_params(model_config, config)

            if config_override:
                for key, value in config_override.items():
                    setattr(config, key, value)

            model = DynamicTransformer(config)
            config=model.config
            model_dir = self.cache_dir / model_name
            model_files = self._find_model_files(model_name, model_dir)

            if not model_files:
                files = self._download_model_files(model_name, 'decoder_only')
                model_files = files.get('model_files', [])
                if 'model_file' in files:
                    model_files = [files['model_file']]
                elif 'index' in files:
                    model_files = [f for f in files.values() if f.suffix in ['.bin', '.safetensors']]

            state_dict = self._load_state_dict_from_files(model_dir, model_files)
            mapped_state_dict = self._map_decoder_weights(state_dict, config, model)

            if self.task_name is not None and self.task_name:
                decoder_name = self.task_name
            else:
                decoder_name = f"causal_{model_name.replace('/', '_').replace('.', '_').replace('-', '_')}"

            task_config = TaskConfig(
                task_name=decoder_name,
                training_strategy=TrainingStrategy.CAUSAL_LM,
                datasets=[],
                text_column="text",
                max_length=config.max_seq_len
            )

            model.add_task(task_config)
            model.decoders[decoder_name].load_state_dict(mapped_state_dict, strict=False)

            print(f"✓ Loaded {model_name} as causal LM task")
            print(f"  Task name: {task_config.task_name}")
            print(f"  Parameters: {sum(p.numel() for p in model.decoders[decoder_name].parameters()):,}")

            return model, task_config

        except Exception as e:
            print(f"Error loading decoder-only model: {e}")
            raise

    def _load_encoder_decoder(self, model_name: str, config: Optional[ModelConfig] = None,
                               config_override: Optional[Dict] = None) -> Tuple[DynamicTransformer, TaskConfig]:

        print("Loading as ENCODER-DECODER → Creating Seq2Seq task")

        try:
            self._download_config(model_name)

            with open(self.cache_dir / model_name / 'config.json', 'r') as f:
                model_config = json.load(f)

            config, _ = self._extract_config_params(model_config, config)

            if config_override:
                for key, value in config_override.items():
                    setattr(config, key, value)

            model = DynamicTransformer(config)
            config=model.config
            model_dir = self.cache_dir / model_name
            model_files = self._find_model_files(model_name, model_dir)

            if not model_files:
                files = self._download_model_files(model_name, 'encoder_decoder')
                model_files = files.get('model_files', [])
                if 'model_file' in files:
                    model_files = [files['model_file']]
                elif 'index' in files:
                    model_files = [f for f in files.values() if f.suffix in ['.bin', '.safetensors']]

            state_dict = self._load_state_dict_from_files(model_dir, model_files)
            encoder_state, decoder_state = self._split_encoder_decoder_weights(state_dict, config)

            model.encoder.load_state_dict(encoder_state, strict=False)

            if self.task_name is not None and self.task_name:
                decoder_name = self.task_name
            else:
                decoder_name = f"seq2seq_{model_name.replace('/', '_').replace('.', '_').replace('-', '_')}"

            task_config = TaskConfig(
                task_name=decoder_name,
                training_strategy=TrainingStrategy.SEQ2SEQ,
                datasets=[],
                source_column="source",
                target_column="target",
                max_length=config.max_seq_len
            )

            model.add_task(task_config)
            model.decoders[decoder_name].load_state_dict(decoder_state, strict=False)

            print(f"✓ Loaded {model_name} as seq2seq task")
            print(f"  Task name: {task_config.task_name}")
            print(f"  Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
            print(f"  Decoder params: {sum(p.numel() for p in model.decoders[decoder_name].parameters()):,}")

            return model, task_config

        except Exception as e:
            print(f"Error loading encoder-decoder model: {e}")
            raise

    def _map_encoder_weights(self, state_dict: Dict, config: ModelConfig, model: DynamicTransformer) -> Dict:
        mapped = {}

        def get_weight(possible_keys, suffix='weight'):
            for key in possible_keys:
                full_key = f"{key}.{suffix}" if suffix else key
                if full_key in state_dict:
                    return full_key, state_dict[full_key]
            return None, None

        try:
            word_emb_keys = ['bert.embeddings.word_embeddings', 'embeddings.word_embeddings', 'wte']
            key, weight = get_weight(word_emb_keys, 'weight')
            if weight is not None:
                mapped['embedding.weight'] = weight
                print(f"  Mapped word embeddings")

            for i in range(config.num_layers):
                target = f'layers.{i}'

                prefixes = [
                    f'bert.encoder.layer.{i}',
                    f'encoder.layer.{i}',
                    f'transformer.encoder.layer.{i}',
                    f'transformer.h.{i}',
                    f'model.layers.{i}'
                ]

                prefix = None
                for p in prefixes:
                    if f'{p}.attention.self.query.weight' in state_dict:
                        prefix = p
                        break
                    if f'{p}.attn.c_attn.weight' in state_dict:
                        prefix = p
                        break

                if prefix is None:
                    continue

                if f'{prefix}.attention.self.query.weight' in state_dict:
                    mapped[f'{target}.attention.q_proj.weight'] = state_dict[f'{prefix}.attention.self.query.weight']
                    mapped[f'{target}.attention.q_proj.bias'] = state_dict[f'{prefix}.attention.self.query.bias']
                    mapped[f'{target}.attention.k_proj.weight'] = state_dict[f'{prefix}.attention.self.key.weight']
                    mapped[f'{target}.attention.k_proj.bias'] = state_dict[f'{prefix}.attention.self.key.bias']
                    mapped[f'{target}.attention.v_proj.weight'] = state_dict[f'{prefix}.attention.self.value.weight']
                    mapped[f'{target}.attention.v_proj.bias'] = state_dict[f'{prefix}.attention.self.value.bias']

                out_proj_keys = [f'{prefix}.attention.output.dense', f'{prefix}.attention.output.Dense']
                key, weight = get_weight(out_proj_keys, 'weight')
                if weight is not None:
                    mapped[f'{target}.attention.out_proj.weight'] = weight
                    _, bias = get_weight(out_proj_keys, 'bias')
                    if bias is not None:
                        mapped[f'{target}.attention.out_proj.bias'] = bias

                attn_norm_keys = [f'{prefix}.attention.output.LayerNorm', f'{prefix}.attention.output.layer_norm']
                key, weight = get_weight(attn_norm_keys, 'weight')
                if weight is not None:
                    mapped[f'{target}.attention_norm.weight'] = weight
                    _, bias = get_weight(attn_norm_keys, 'bias')
                    if bias is not None:
                        mapped[f'{target}.attention_norm.bias'] = bias

                ffn_keys = [f'{prefix}.intermediate.dense', f'{prefix}.ffn.dense']
                key, weight = get_weight(ffn_keys, 'weight')
                if weight is not None:
                    mapped[f'{target}.ffn.gate_proj.weight'] = weight
                    mapped[f'{target}.ffn.up_proj.weight'] = weight
                    _, bias = get_weight(ffn_keys, 'bias')
                    if bias is not None:
                        mapped[f'{target}.ffn.gate_proj.bias'] = bias
                        mapped[f'{target}.ffn.up_proj.bias'] = bias

                out_ffn_keys = [f'{prefix}.output.dense', f'{prefix}.ffn.output']
                key, weight = get_weight(out_ffn_keys, 'weight')
                if weight is not None:
                    mapped[f'{target}.ffn.down_proj.weight'] = weight
                    _, bias = get_weight(out_ffn_keys, 'bias')
                    if bias is not None:
                        mapped[f'{target}.ffn.down_proj.bias'] = bias

                ffn_norm_keys = [f'{prefix}.output.LayerNorm', f'{prefix}.output.layer_norm']
                key, weight = get_weight(ffn_norm_keys, 'weight')
                if weight is not None:
                    mapped[f'{target}.ffn_norm.weight'] = weight
                    _, bias = get_weight(ffn_norm_keys, 'bias')
                    if bias is not None:
                        mapped[f'{target}.ffn_norm.bias'] = bias

                if i < 2:
                    print(f"  Mapped layer {i}")

            final_norm_keys = [
                'bert.encoder.LayerNorm', 'bert.encoder.layer_norm',
                'encoder.LayerNorm', 'transformer.encoder.LayerNorm', 'ln_f'
            ]
            key, weight = get_weight(final_norm_keys, 'weight')
            if weight is not None:
                mapped['norm.weight'] = weight
                _, bias = get_weight(final_norm_keys, 'bias')
                if bias is not None:
                    mapped['norm.bias'] = bias
                print(f"  Mapped final layer norm")
            else:
                mapped['norm.weight'] = torch.ones(config.embed_dim)
                mapped['norm.bias'] = torch.zeros(config.embed_dim)
                print(f"  Initialized final layer norm with identity")

        except Exception as e:
            print(f"Warning: Error during weight mapping: {e}")

        return mapped
    def _map_decoder_weights(self, state_dict: Dict, config: ModelConfig, model: DynamicTransformer) -> Dict:
        mapped = {}
    
        try:
            if 'transformer.wte.weight' in state_dict:
                src_weight = state_dict['transformer.wte.weight']
                target_vocab_size = config.vocab_size
                src_vocab_size = src_weight.shape[0]
                
                if src_vocab_size != target_vocab_size:
                    print(f"  Vocabulary size mismatch: checkpoint has {src_vocab_size}, model expects {target_vocab_size}")
                    if src_vocab_size > target_vocab_size:
                        mapped['embedding.weight'] = src_weight[:target_vocab_size]
                        print(f"  Truncated embedding weights from {src_vocab_size} to {target_vocab_size}")
                    else:
                        from torch import nn
                        padded = nn.Parameter(torch.zeros(target_vocab_size, src_weight.shape[1]))
                        padded[:src_vocab_size] = src_weight
                        mapped['embedding.weight'] = padded
                        print(f"  Padded embedding weights from {src_vocab_size} to {target_vocab_size}")
                else:
                    mapped['embedding.weight'] = src_weight
                print(f"  Mapped token embeddings: shape {mapped['embedding.weight'].shape}")
    
            for i in range(config.num_layers):
                prefixes = [f'transformer.h.{i}', f'model.layers.{i}']
                prefix = None
                for p in prefixes:
                    if f'{p}.attn.c_attn.weight' in state_dict:
                        prefix = p
                        break
    
                if prefix is None:
                    continue
    
                target = f'layers.{i}'
    
                combined_weight = state_dict[f'{prefix}.attn.c_attn.weight']
                combined_bias = state_dict[f'{prefix}.attn.c_attn.bias']
    
                embed_dim = combined_weight.shape[0] // 3
    
                if embed_dim > 0:
                    q_weight = combined_weight[:embed_dim]
                    k_weight = combined_weight[embed_dim:2*embed_dim]
                    v_weight = combined_weight[2*embed_dim:]
    
                    q_bias = combined_bias[:embed_dim]
                    k_bias = combined_bias[embed_dim:2*embed_dim]
                    v_bias = combined_bias[2*embed_dim:]
    
                    mapped[f'{target}.self_attention.q_proj.weight'] = q_weight
                    mapped[f'{target}.self_attention.k_proj.weight'] = k_weight
                    mapped[f'{target}.self_attention.v_proj.weight'] = v_weight
                    mapped[f'{target}.self_attention.q_proj.bias'] = q_bias
                    mapped[f'{target}.self_attention.k_proj.bias'] = k_bias
                    mapped[f'{target}.self_attention.v_proj.bias'] = v_bias
    
                if f'{prefix}.attn.c_proj.weight' in state_dict:
                    mapped[f'{target}.self_attention.out_proj.weight'] = state_dict[f'{prefix}.attn.c_proj.weight']
                    mapped[f'{target}.self_attention.out_proj.bias'] = state_dict[f'{prefix}.attn.c_proj.bias']
    
                if f'{prefix}.ln_1.weight' in state_dict:
                    mapped[f'{target}.norm1.weight'] = state_dict[f'{prefix}.ln_1.weight']
                    mapped[f'{target}.norm1.bias'] = state_dict[f'{prefix}.ln_1.bias']
    
                if f'{prefix}.mlp.c_fc.weight' in state_dict:
                    mapped[f'{target}.ffn.gate_proj.weight'] = state_dict[f'{prefix}.mlp.c_fc.weight']
                    mapped[f'{target}.ffn.gate_proj.bias'] = state_dict[f'{prefix}.mlp.c_fc.bias']
                    mapped[f'{target}.ffn.up_proj.weight'] = state_dict[f'{prefix}.mlp.c_fc.weight']
                    mapped[f'{target}.ffn.up_proj.bias'] = state_dict[f'{prefix}.mlp.c_fc.bias']
    
                if f'{prefix}.mlp.c_proj.weight' in state_dict:
                    mapped[f'{target}.ffn.down_proj.weight'] = state_dict[f'{prefix}.mlp.c_proj.weight']
                    mapped[f'{target}.ffn.down_proj.bias'] = state_dict[f'{prefix}.mlp.c_proj.bias']
    
                if f'{prefix}.ln_2.weight' in state_dict:
                    mapped[f'{target}.norm3.weight'] = state_dict[f'{prefix}.ln_2.weight']
                    mapped[f'{target}.norm3.bias'] = state_dict[f'{prefix}.ln_2.bias']
    
                if i < 2:
                    print(f"  Mapped layer {i}")
    
            if 'transformer.ln_f.weight' in state_dict:
                mapped['norm.weight'] = state_dict['transformer.ln_f.weight']
                mapped['norm.bias'] = state_dict['transformer.ln_f.bias']
                print(f"  Mapped final layer norm")
    
            if 'lm_head.weight' in state_dict:
                src_weight = state_dict['lm_head.weight']
                target_vocab_size = config.vocab_size
                src_vocab_size = src_weight.shape[0]
                
                if src_vocab_size != target_vocab_size:
                    if src_vocab_size > target_vocab_size:
                        mapped['output_proj.weight'] = src_weight[:target_vocab_size]
                        print(f"  Truncated output projection from {src_vocab_size} to {target_vocab_size}")
                    else:
                        from torch import nn
                        padded = nn.Parameter(torch.zeros(target_vocab_size, src_weight.shape[1]))
                        padded[:src_vocab_size] = src_weight
                        mapped['output_proj.weight'] = padded
                        print(f"  Padded output projection from {src_vocab_size} to {target_vocab_size}")
                else:
                    mapped['output_proj.weight'] = src_weight
                print("  Mapped output projection")
            elif 'transformer.wte.weight' in state_dict:
                src_weight = state_dict['transformer.wte.weight']
                target_vocab_size = config.vocab_size
                src_vocab_size = src_weight.shape[0]
                
                if src_vocab_size != target_vocab_size:
                    if src_vocab_size > target_vocab_size:
                        mapped['output_proj.weight'] = src_weight[:target_vocab_size]
                        print(f"  Truncated output projection from {src_vocab_size} to {target_vocab_size}")
                    else:
                        from torch import nn
                        padded = nn.Parameter(torch.zeros(target_vocab_size, src_weight.shape[1]))
                        padded[:src_vocab_size] = src_weight
                        mapped['output_proj.weight'] = padded
                        print(f"  Padded output projection from {src_vocab_size} to {target_vocab_size}")
                else:
                    mapped['output_proj.weight'] = src_weight
                print(f"  Mapped output projection (tied with embeddings)")
    
        except Exception as e:
            print(f"Warning: Error during decoder weight mapping: {e}")
    
        return mapped
    

    def _split_encoder_decoder_weights(self, state_dict: Dict, config: ModelConfig) -> Tuple[Dict, Dict]:
        encoder_state = {}
        decoder_state = {}

        try:
            for key, value in state_dict.items():
                if 'encoder' in key:
                    new_key = key.replace('encoder.', '')
                    encoder_state[new_key] = value
                elif 'decoder' in key:
                    new_key = key.replace('decoder.', '')
                    decoder_state[new_key] = value
                elif 'shared' in key:
                    encoder_state[key] = value
                    decoder_state[key] = value
                elif 'lm_head' in key:
                    decoder_state[key] = value
                else:
                    if 'bert' in key or 'roberta' in key:
                        encoder_state[key] = value
                    elif 'gpt' in key or 'llama' in key:
                        decoder_state[key] = value
                    else:
                        encoder_state[key] = value
                        decoder_state[key] = value

        except Exception as e:
            print(f"Warning: Error splitting weights: {e}")

        return encoder_state, decoder_state

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        self.model_name = model_name

        try:
            self._download_config(model_name)

            model_dir = self.cache_dir / model_name
            config_path = model_dir / 'config.json'
            self.vocab_path=self._download_tokenizer_files(model_name,model_dir)

            if not config_path.exists():
                return {"error": f"Could not find config for {model_name}"}

            with open(config_path, 'r') as f:
                full_config = json.load(f)

            architecture = self._detect_architecture(model_name)

            _, params = self._extract_config_params(full_config)

            info = {
                "model_name": model_name,
                "architecture": architecture,
                "embed_dim": params['embed_dim'],
                "num_layers": params['num_layers'],
                "num_heads": params['num_heads'],
                "vocab_size": params['vocab_size'],
                "max_seq_len": params['max_seq_len'],
                "dropout": params['dropout'],
                "ff_mult": params['ff_mult'],
                "model_type": params['model_type'],
                'suport_simpletokenizer': params['suport_simpletokenizer'],
            }

            return info

        except Exception as e:
            return {"error": str(e)}
    def _download_tokenizer_files(self, model_name: str, model_dir: Path) -> str:
        tokenizer_files = [
            'vocab.txt',
            'vocab.json',
            'tokenizer.json',
        ]
        
        vocab_path = model_dir / 'vocab.json'
        
        if vocab_path.exists():
            print(f"Using cached vocab.json from {vocab_path}")
            with open(str(vocab_path), 'r', encoding='utf-8') as f:
                vocab_dict = json.load(f)
            self.vocab_size=len(vocab_dict)
            return str(vocab_path)
        
        for tokenizer_file in tokenizer_files:
            url = f"https://huggingface.co/{model_name}/resolve/main/{tokenizer_file}"
            filepath = model_dir / tokenizer_file
            
            try:
                response = requests.head(url, timeout=10)
                if response.status_code != 200:
                    continue
                
                print(f"Downloading {tokenizer_file}...")
                self._download_file(url, filepath)
                
                if tokenizer_file == 'vocab.txt':
                    with open(str(filepath), 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        vocab_dict = {token: idx for idx, token in enumerate(lines)}
                    
                    with open(str(vocab_path), 'w', encoding='utf-8') as f:
                        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
                    print(f"Built vocab.json from vocab.txt ({len(vocab_dict)} tokens)")
                    self.vocab_size=len(vocab_dict)
                    return str(vocab_path)
                
                elif tokenizer_file == 'vocab.json':
                    with open(str(filepath), 'r', encoding='utf-8') as f:
                        vocab_dict = json.load(f)
                    print(f"Loaded vocab.json ({len(vocab_dict)} tokens)")
                    self.vocab_size=len(vocab_dict)
                    return str(filepath)
                
                elif tokenizer_file == 'tokenizer.json':
                    with open(str(filepath), 'r', encoding='utf-8') as f:
                        tokenizer_data = json.load(f)
                    
                    if 'model' in tokenizer_data and 'vocab' in tokenizer_data['model']:
                        vocab_dict = tokenizer_data['model']['vocab']
                        with open(str(vocab_path), 'w', encoding='utf-8') as f:
                            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
                        print(f"Built vocab.json from tokenizer.json ({len(vocab_dict)} tokens)")
                        self.vocab_size=len(vocab_dict)
                        return str(vocab_path)
                        
            except Exception as e:
                #print(f"Warning: Failed to process {tokenizer_file}: {e}")
                continue
        
        print("Warning: No supported tokenizer file found. SimpleTokenizer will use default vocab.")
        return None
    def _extract_config_params(
        self,
        model_config: Dict[str, Any],
        dytr_config: Optional[ModelConfig] = None
    ) -> Tuple[ModelConfig, Dict[str, Any]]:

        embed_dim = (
            model_config.get('hidden_size') or
            model_config.get('d_model') or
            model_config.get('n_embd') or
            model_config.get('dim') or
            model_config.get('embed_dim') or
            model_config.get('embedding_dim') or
            model_config.get('hidden_dim') or
            768
        )

        num_layers = (
            model_config.get('num_hidden_layers') or
            model_config.get('num_layers') or
            model_config.get('n_layer') or
            model_config.get('encoder_layers') or
            model_config.get('num_blocks') or
            12
        )

        num_heads = (
            model_config.get('num_attention_heads') or
            model_config.get('num_heads') or
            model_config.get('n_head') or
            model_config.get('attention_heads') or
            model_config.get('num_attn_heads') or
            12
        )

        max_seq_len = (
            model_config.get('max_position_embeddings') or
            model_config.get('n_positions') or
            model_config.get('max_seq_len') or
            model_config.get('max_length') or
            model_config.get('max_sequence_length') or
            512
        )

        dropout = (
            model_config.get('hidden_dropout_prob') or
            model_config.get('resid_pdrop') or
            model_config.get('dropout') or
            model_config.get('attention_dropout') or
            model_config.get('dropout_rate') or
            0.1
        )

        vocab_size = (
            model_config.get('vocab_size') or
            model_config.get('vocab_size_vocab') or
            model_config.get('vocab') or
            model_config.get('vocab_source') or 
            30522
        )

        ff_dim = (
            model_config.get('intermediate_size') or
            model_config.get('ffn_dim') or
            model_config.get('d_ff') or
            model_config.get('encoder_ffn_dim') or
            model_config.get('decoder_ffn_dim') or
            embed_dim * 4
        )

        if isinstance(ff_dim, int) and isinstance(embed_dim, int) and embed_dim > 0:
            ff_mult = ff_dim // embed_dim
        else:
            ff_mult = 4

        model_type = model_config.get('model_type', 'unknown')

        is_decoder_only = model_type in ['gpt2', 'gpt', 'llama', 'mistral', 'phi', 'gemma', 'bloom', 'opt', 'falcon', 'qwen']

        params = {
            'embed_dim': embed_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'head_dim': embed_dim // num_heads if num_heads > 0 else 64,
            'ff_mult': ff_mult,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
            'vocab_size':self.vocab_size if self.vocab_size and self.vocab_size >1 else vocab_size,
            'ff_dim': ff_dim,
            'model_type': model_type,
            'is_decoder_only': is_decoder_only,
            'suport_simpletokenizer': True if self.vocab_path else False
        }

        if dytr_config is None:
            dytr_config = ModelConfig(
                embed_dim=params['embed_dim'],
                num_layers=params['num_layers'],
                num_heads=params['num_heads'],
                head_dim=params['head_dim'],
                ff_mult=params['ff_mult'],
                max_seq_len=params['max_seq_len'],
                dropout=params['dropout'],
                
                use_task_adapters=True if is_decoder_only else False,
                use_rotary_embedding=False,
                vocab_size=params['vocab_size'],
                
                special_tokens={},
                tokenizer_name=self.model_name,
                
            )
        else:
            dytr_config.embed_dim = params['embed_dim']
            dytr_config.num_layers = params['num_layers']
            dytr_config.num_heads = params['num_heads']
            dytr_config.head_dim = params['head_dim']
            dytr_config.ff_mult = params['ff_mult']
            dytr_config.max_seq_len = params['max_seq_len']
            dytr_config.dropout = params['dropout']
            dytr_config.vocab_size=params['vocab_size']
            dytr_config.special_tokens={}
            dytr_config.tokenizer_name=self.model_name
        if dytr_config.use_simple_tokenizer and self.vocab_path is not None:
            dytr_config.tokenizer_name=self.vocab_path
            dytr_config.use_simple_tokenizer=True
            dytr_config.training_from_scratch=False
        else:
            dytr_config.use_simple_tokenizer=False
            dytr_config.training_from_scratch=False

        return dytr_config, params
