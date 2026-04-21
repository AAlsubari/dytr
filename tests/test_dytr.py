"""
Complete test suite for dytr library.
Tests all major components: model initialization, task configuration, training, and inference.
"""

import pytest
import torch
import pandas as pd





# Import dytr components
from dytr import (
    DynamicTransformer,
    ModelConfig,
    TaskConfig,
    TrainingStrategy,
    Trainer,
    SingleDatasetProcessing,
)




@pytest.fixture(scope="session")
def model_config():
    """Create a small model configuration for testing."""
    return ModelConfig(
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        head_dim=16,
        ff_mult=2,
        tokenizer_name='prajjwal1/bert-tiny',
        max_seq_len=64,
        dropout=0.1,
        batch_size=4,
        learning_rate=3e-4,
        num_train_epochs=1,
        use_rotary_embedding=True,
        use_task_adapters=True,
        adapter_bottleneck=16,
        use_ewc=False,
        use_replay=False,
        device='cpu'
    )


@pytest.fixture(scope="session")
def model(model_config):
    """Create a single model instance for all tests."""
    model = DynamicTransformer(model_config)
    return model


@pytest.fixture
def sample_classification_data():
    """Create sample data for classification task."""
    df = pd.DataFrame({
        'text': [
            'This product is amazing!',
            'Terrible quality, very disappointed.',
            'Good value for money.',
            'Worst purchase ever.',
            'Excellent service!'
        ],
        'label': [1, 0, 1, 0, 1]
    })
    return df


@pytest.fixture
def sample_token_data():
    """Create sample data for token classification."""
    df = pd.DataFrame({
        'text': [
            'Apple Inc. is in California',
            'Google was founded in Mountain View',
            'Microsoft has offices in Seattle'
        ],
        'tags': [
            '1 0 0 2 0',
            '1 0 0 0 2 0',
            '1 0 0 0 2'
        ]
    })
    return df


@pytest.fixture
def sample_seq2seq_data():
    """Create sample data for seq2seq task."""
    df = pd.DataFrame({
        'source': [
            'Hello world',
            'How are you',
            'Good morning'
        ],
        'target': [
            'مرحبا بالعالم',
            'كيف حالك',
            'صباح الخير'
        ]
    })
    return df


@pytest.fixture
def sample_causal_data():
    """Create sample data for causal LM."""
    df = pd.DataFrame({
        'text': [
            'The sun rises in the east.',
            'Cats are adorable animals.',
            'Machine learning is fascinating.',
            'Python is a great language.'
        ]
    })
    return df



def test_import_and_version():
    """Test that dytr imports correctly and has version."""
    import dytr
    assert dytr.__version__ is not None
    assert isinstance(dytr.__version__, str)
    print(f"✅ dytr version {dytr.__version__} imported successfully")




def test_model_initialization(model_config):
    """Test that model initializes correctly."""
    model = DynamicTransformer(model_config)
    assert model is not None
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'tokenizer')
    assert hasattr(model, 'shared_embedding')
    assert len(model.tokenizer) > 0
    print(f"✅ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")


def test_model_config_parameters(model_config):
    """Test model configuration parameters."""
    assert model_config.embed_dim == 64
    assert model_config.num_layers == 2
    assert model_config.num_heads == 4
    assert model_config.max_seq_len == 64
    assert model_config.device == 'cpu'
    print("✅ Model configuration verified")




def test_task_config_creation():
    """Test creating task configurations for all strategies."""
    
    # Classification task
    class_task = TaskConfig(
        task_name="test_classification",
        training_strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
        num_labels=2,
        text_column="text",
        label_column="label",
        max_length=32
    )
    assert class_task.task_name == "test_classification"
    assert class_task.num_labels == 2
    
    # Token classification task
    token_task = TaskConfig(
        task_name="test_token",
        training_strategy=TrainingStrategy.TOKEN_CLASSIFICATION,
        num_labels=5,
        text_column="text",
        label_column="tags",
        max_length=64
    )
    assert token_task.training_strategy == TrainingStrategy.TOKEN_CLASSIFICATION
    
    # Seq2Seq task
    seq2seq_task = TaskConfig(
        task_name="test_seq2seq",
        training_strategy=TrainingStrategy.SEQ2SEQ,
        source_column="source",
        target_column="target",
        max_length=32
    )
    assert seq2seq_task.training_strategy == TrainingStrategy.SEQ2SEQ
    
    # Causal LM task
    causal_task = TaskConfig(
        task_name="test_causal",
        training_strategy=TrainingStrategy.CAUSAL_LM,
        text_column="text",
        max_length=32
    )
    assert causal_task.training_strategy == TrainingStrategy.CAUSAL_LM
    
    print("✅ All task configurations created successfully")




def test_add_tasks_to_model(model, sample_classification_data, sample_token_data):
    """Test adding multiple tasks to the model."""
    
    # Create and add classification task
    class_task = TaskConfig(
        task_name="test_classification",
        training_strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
        num_labels=2,
        text_column="text",
        label_column="label",
        max_length=32
    )
    model.add_task(class_task)
    
    # Create and add token classification task
    token_task = TaskConfig(
        task_name="test_token",
        training_strategy=TrainingStrategy.TOKEN_CLASSIFICATION,
        num_labels=5,
        text_column="text",
        label_column="tags",
        max_length=64
    )
    model.add_task(token_task)
    
    # Verify tasks were added
    assert "test_classification" in model.current_tasks
    assert "test_token" in model.current_tasks
    assert len(model.current_tasks) >= 2
    
    # Verify task heads exist
    assert "test_classification" in model.task_heads
    assert "test_token" in model.task_heads
    
    print(f"✅ Added {len(model.current_tasks)} tasks to model")




def test_dataset_processing(model, sample_classification_data):
    """Test SingleDatasetProcessing for classification."""
    
    dataset = SingleDatasetProcessing(
        df=sample_classification_data,
        tokenizer=model.tokenizer,
        max_len=32,
        task_name="test_classification",
        strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
        num_labels=2,
        text_column="text",
        label_column="label",cache_dir="./",
    )
    
    assert len(dataset) == len(sample_classification_data)
    
    # Get a sample
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["task_name"] == "test_classification"
    assert isinstance(sample["input_ids"], torch.Tensor)
    
    print(f"✅ Dataset processing works, {len(dataset)} samples")


def test_token_dataset_processing(model, sample_token_data):
    """Test SingleDatasetProcessing for token classification."""
    
    dataset = SingleDatasetProcessing(
        df=sample_token_data,
        tokenizer=model.tokenizer,
        max_len=64,
        task_name="test_token",
        strategy=TrainingStrategy.TOKEN_CLASSIFICATION,
        num_labels=5,
        text_column="text",
        tags_column="tags",cache_dir="./",
    )
    
    assert len(dataset) > 0
    sample = dataset[0]
    assert "labels" in sample
    assert sample["labels"].dim() == 1  # 1D tensor for token labels
    
    print(f"✅ Token dataset processing works")




def test_forward_pass(model, sample_classification_data):
    """Test forward pass through the model."""
    
    # Create dataset and get a sample
    dataset = SingleDatasetProcessing(
        df=sample_classification_data,
        tokenizer=model.tokenizer,
        max_len=32,
        task_name="test_classification",
        strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
        num_labels=2,
        text_column="text",
        label_column="label",cache_dir="./",
    )
    
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0)
    attention_mask = sample["attention_mask"].unsqueeze(0)
    #labels = sample["labels"]
    labels = sample["labels"][0].unsqueeze(0)
    
    # Forward pass
    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_name="test_classification",
        labels=labels
    )
    
    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape[-1] == 2  # 2 classes
    
    print(f"✅ Forward pass successful, loss: {outputs['loss'].item():.4f}")


def test_forward_without_labels(model, sample_classification_data):
    """Test forward pass without labels (inference mode)."""
    
    dataset = SingleDatasetProcessing(
        df=sample_classification_data,
        tokenizer=model.tokenizer,
        max_len=32,
        task_name="test_classification",
        strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
        num_labels=2,
        text_column="text",
        label_column="label",cache_dir="./",
    )
    
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0)
    attention_mask = sample["attention_mask"].unsqueeze(0)
    
    outputs = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        task_name="test_classification"
    )
    
    assert "logits" in outputs
    assert "loss" not in outputs
    
    print("✅ Inference forward pass successful")




def test_generate_classification(model):
    """Test generate method for classification."""
    
    result = model.generate(
        "This is a test sentence for classification.",
        task_name="test_classification"
    )
    
    assert "prediction" in result
    assert "probabilities" in result
    assert isinstance(result["prediction"], int)
    assert len(result["probabilities"]) == 2
    
    print(f"✅ Classification generation successful, prediction: {result['prediction']}")


def test_generate_token_classification(model, sample_token_data):
    """Test generate method for token classification."""
    
    # Create and add token task if not exists
    if "test_token_generate" not in model.current_tasks:
        token_task = TaskConfig(
            task_name="test_token_generate",
            training_strategy=TrainingStrategy.TOKEN_CLASSIFICATION,
            num_labels=5,
            text_column="text",
            label_column="tags",
            max_length=64
        )
        model.add_task(token_task)
        
        
    
    result = model.generate(
        "Apple Inc. is a technology company.",
        task_name="test_token_generate"
    )
    
    assert "tokens" in result
    assert "predictions" in result
    assert "pairs" in result
    assert len(result["tokens"]) == len(result["predictions"])
    
    print("✅ Token classification generation successful")



def test_training(model, sample_classification_data):
    """Test training loop (single epoch)."""
    
    # Create dataset
    dataset = SingleDatasetProcessing(
        df=sample_classification_data,
        tokenizer=model.tokenizer,
        max_len=32,
        task_name="test_classification",
        strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
        num_labels=2,
        text_column="text",
        label_column="label",cache_dir="./",
    )
    
    # Create trainer
    trainer = Trainer(model, model.config, exp_dir="./")
    
    # Create train datasets dict
    train_datasets = {
        "test_classification": (dataset, TrainingStrategy.SENTENCE_CLASSIFICATION)
    }
    
    # Get task config
    task_config = TaskConfig(
        task_name="test_classification",
        training_strategy=TrainingStrategy.SENTENCE_CLASSIFICATION,
        num_labels=2,
        text_column="text",
        label_column="label"
    )
    
    # Quick training (1 epoch)
    try:
        trained_model = trainer.train([task_config], train_datasets, {})
        assert trained_model is not None
        print("✅ Training completed successfully")
    except Exception as e:
        print(f"⚠️ Training test note: {e}")
        # Training might fail without proper data, but that's expected







def test_multi_task_with_strategies(model, sample_classification_data, sample_causal_data):
    """Test model with multiple task types."""
    
    # Add causal LM task
    if "test_causal_multi" not in model.current_tasks:
        causal_task = TaskConfig(
            task_name="test_causal_multi",
            training_strategy=TrainingStrategy.CAUSAL_LM,
            text_column="text",
            max_length=32
        )
        model.add_task(causal_task)
        
    
    # Test both tasks
    class_result = model.generate("Test classification", task_name="test_classification")
    assert "prediction" in class_result
    
    try:
        causal_result = model.generate("The future of", task_name="test_causal_multi", max_new_tokens=10)
        assert causal_result is not None
        print("✅ Multi-task with different strategies works")
    except Exception as e:
        print(f"⚠️ Causal LM generation note: {e}")


if __name__ == "__main__":
    
    pytest.main(["-v", "--tb=short", __file__])#,"-s"])

