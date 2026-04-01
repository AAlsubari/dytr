# dytr - Dynamic Transformer Library
dytr is a flexible PyTorch library for multi-task learning with dynamic transformer architectures. Train multiple tasks sequentially or simultaneously while preserving performance on previous tasks through built-in continual learning techniques. it also supports to finetune and modify pretrained model such as bert.




[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://badge.fury.io/py/dytr.svg)](https://badge.fury.io/py/dytr)

**Build dynamic transformers that learn multiple tasks.**
## Why dytr?

- 🎯 **Multi-Task Ready** - Train classification, generation, and sequence tasks in one model
- 🧠 **Never Forgets** - Built-in EWC and experience replay prevent catastrophic forgetting
- 🔧 **No Black Box** - Full control over architecture, understand every component
- ⚡ **Lightweight** - Pure PyTorch, minimal dependencies
- 📦 **Pretrained Support** - Load BERT, RoBERTa, and more as your encoder backbone and fine tune it on multiple tasks.
- 

## Installation

```bash
pip install dytr
```

## Quick Start

```python
from dytr import DynamicTransformer, ModelConfig, TaskConfig, TrainingStrategy, Trainer SingleDatasetProcessing
import pandas as pd

# 1. Configure your transformer
config = ModelConfig(
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    max_seq_len=256
)

# 2. Create the model
model = DynamicTransformer(config)

# data loading and processing
train_data = pd.DataFrame({
    'text': ['Great movie!', 'Terrible film.', 'Amazing acting!', 'Boring plot.'],
    'label': [1, 0, 1, 0]
})
train_dataset = SingleDatasetProcessing(
    df=train_data,
    tokenizer=model.tokenizer,
    max_len=128,
    task_name="sentiment_analysis",
    strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
    text_column="text",
    label_column="label"
)
# 3. Add a task
task = TaskConfig(
    task_name="sentiment_analysis",
    training_strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
    num_labels=2,# train_data.num_labels
)
#model.add_task(task) # not require it will be add automatically during the training process

# Initialize trainer and train
trainer = Trainer(model, config, exp_dir="./experiments")
train_datasets = {"sentiment_analysis": (train_dataset, TrainingStrategy.SENTENCE_CLASSIFICATION)}
model = trainer.train([classification_task], train_datasets, {})# you can set more than one for list of tasks and dataset for multitasks training 

# 4. Generate predictions
result = model.generate("This product is amazing!", task_name="sentiment_analysis")
print(f"Prediction: {result['prediction']}")

# Save the entire multi-task model
model.save_model("multi_task_model.pt")

# Load the model
loaded_model = DynamicTransformer.load_model("multi_task_model.pt")


```

## Core Capabilities

### Multiple Training Strategies

| Strategy | Purpose | Use Case |
|----------|---------|----------|
| **Causal LM** | Autoregressive text generation | Chatbots, content creation |
| **Seq2Seq** | Input to output transformation | Translation, summarization |
| **Sentence Classification** | Document-level categorization | Sentiment, topic detection |
| **Token Classification** | Token-level labeling | Named entity recognition, POS tagging |

### Continual Learning

Train tasks sequentially without losing previous knowledge:

```python
config = ModelConfig(
    use_ewc=True,              # Protect important weights
    use_replay=True,           # Replay old samples
    use_task_adapters=True,    # Task-specific modules
    ewc_lambda=1000.0,
    replay_buffer_size=2000
)

model = DynamicTransformer(config)

# Train tasks one after another
for task in task_list:
    model.add_task(task)
    trainer.train([task], train_data, val_data)
    # Previous tasks remain accurate
    # The trainer automatically handles EWC and replay buffer, but you should add the samples to the pretrained model
```

### Pretrained Encoders

Load powerful encoders as your backbone and extend them with tasks:

```python
from dytr import PretrainedModelLoader

loader = PretrainedModelLoader()
config = ModelConfig(tokenizer_name='bert-base-uncased',per_device_train_batch_size=32,num_train_epochs=3,per_device_eval_batch_size=8,special_tokens={},use_task_adapters=False,use_ewc=True,use_replay=True,use_rotary_embedding=False, training_from_scratch=False)

# Load pretrained BERT as your encoder
model = loader.load_pretrained('bert-base-uncased', config)

# Now add your own tasks - the model is fully dytr compatible
class_train = pd.DataFrame(
        {
            "text": [
                "Great product!",
                "Poor quality.",
                "Excellent service!",
                "Very disappointed.",
                "Highly recommended!",
            ],
            "label": [1, 0, 1, 0, 1],
        }
    )
classification_task = TaskConfig(
        task_name="sentiment",
        training_strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
        num_labels=2,
        text_column="text",
        label_column="label",
        max_length=128,
    )
class_dataset = SingleDatasetProcessing(
        df=class_train,
        tokenizer=model.tokenizer,
        max_len=classification_task.max_length,
        task_name=classification_task.task_name,
        strategy=classification_task.training_strategy,
        num_labels=classification_task.num_labels,
        text_column=classification_task.text_column,
        label_column=classification_task.label_column,
    )
# Causal LM task data (text generation)
lm_train = pd.DataFrame(
        {
            "text": [
                "The sun rises in the east.",
                "Cats are adorable animals.",
                "Machine learning is fascinating.",
                "Python is a great programming language.",
                "Deep learning powers modern AI.",
            ]
        }
    )
lm_task = TaskConfig(
        task_name="text_generation",
        training_strategy=TrainingStrategy.CAUSAL_LM,
        max_length=256,
    )
lm_dataset = SingleDatasetProcessing(
        df=lm_train,
        tokenizer=model.tokenizer,
        max_len=lm_task.max_length,
        task_name=lm_task.task_name,
        strategy=lm_task.training_strategy,
        text_column="text",
    )
train_datasets = {
        classification_task.task_name: (class_dataset, classification_task.training_strategy),
        lm_task.task_name: (lm_dataset, lm_task.training_strategy),
    }

    val_datasets = {
        #classification_task.task_name: (class_val_dataset, classification_task.training_strategy)
    }

# 6. Train model
print("\n6. Training model...")
trainer = Trainer(model, config, exp_dir="./multi_task_experiments")
model = trainer.train([classification_task, lm_task], train_datasets, val_datasets)

#model = trainer.train([ lm_task], train_datasets, val_datasets)
test_texts = ["This is amazing!", "I hate this."]
for text in test_texts:
      result = model.generate(text, task_name="sentiment")
      sentiment = "POSITIVE" if result["prediction"] == 1 else "NEGATIVE"
      print(f"      {text} -> {sentiment}")

# Test generation
print("\n   Text generation test:")
prompt = "The future of technology"
generated = model.generate(prompt, task_name="text_generation", max_new_tokens=20)
print(f"      Prompt: {prompt}")
print(f"      Generated: {generated}")



#model.add_task(sentiment_task)
#model.add_task(ner_task)
#model.add_task(translation_task)

# Train, generate, and use just like any dytr model
```

### Task-Specific Learning Rates

Different components learn at different speeds:

```python
config = ModelConfig(
    learning_rate=3e-4,
    head_lr_mult=2.0,      # Task heads: fast adaptation
    decoder_lr_mult=0.5,   # Decoders: moderate
    shared_lr_mult=0.1     # Shared encoder: preserve knowledge
)
```

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         DynamicTransformer              │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐   │
│  │     Shared Encoder               │   │
│  │  (Pretrained or from scratch)    │   │
│  └─────────────────────────────────┘   │
│                  │                      │
│    ┌─────────────┼─────────────┐        │
│    ▼             ▼             ▼        │
│ ┌──────┐    ┌──────┐    ┌──────┐       │
│ │Task 1│    │Task 2│    │Task 3│       │
│ │ Head │    │ Head │    │Decoder│      │
│ └──────┘    └──────┘    └──────┘       │
│    │           │           │            │
│    ▼           ▼           ▼            │
│Classification  NER    Generation       │
└─────────────────────────────────────────┘
```



## Who Should Use dytr?

| Audience | Why It Matters |
|----------|----------------|
| **Researchers** | Customize every aspect of the transformer architecture, Test continual learning algorithms with EWC and experience replay, experiment with multi-task architectures, Experiment with task-specific learning rates and adapters, Analyze forgetting behavior across sequential tasks |
| **Developers** | Add new tasks without retraining from scratch, Load pretrained models and extend them with your own tasks, Build production-ready multi-task systems without complex dependencies |
| **Students** | Understand transformers from scratch with transparent, readable code, Visualize the impact of hyperparameters on model size, Learn multi-task learning concepts hands-on |
| **Organizations** | Deploy single models that handle multiple tasks efficiently , Deploy lighter, faster inference systems, Maintain knowledge across task updates with continual learning |

## Key Differentiators

- **Full Transparency** - No hidden complexity, understand every component
- **Continual Learning First** - Built from the ground up for sequential task learning
- **Truly Dynamic** - Add or remove tasks without retraining from scratch
- **Pure PyTorch** - No heavy dependencies, easy to customize

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- pandas
- scikit-learn
- tqdm
- requests

## Documentation

- **ModelConfig**: Architecture, training, and continual learning parameters
- **TaskConfig**: Dataset configuration, column mapping, task-specific settings
- **TrainingStrategy**: Causal LM, Seq2Seq, Sentence Classification, Token Classification
- **PretrainedModelLoader**: Load BERT, RoBERTa, DistilBERT, ALBERT as encoders

## License

Apache License 2.0

## Author

**Dr. Akram Alsubari**


## Contributing

Contributions are welcome! Open issues or share your use cases.
## Support and Contact
For questions, issues, or suggestions:
For questions, issues, or suggestions:
- 📧 **Email**: akram.alsubari@outlook.com
- 🔗 **LinkedIn**: [https://www.linkedin.com/in/akram-alsubari/](https://www.linkedin.com/in/akram-alsubari/)
- 📱 **Connect**: Feel free to reach out for collaborations, research discussions, or feedback

- 🎓 **Research Interests**: Natural Language Processing, Deep Learning, Transformers, Continual Learning, Multi-Task Learning, Large Language Models
---

**Build once. Learn multiple tasks. Never forget.**
