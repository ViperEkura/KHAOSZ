import os
import json
import torch
import shutil
import pytest
import tempfile
import safetensors.torch as st
from khaosz.trainer import *
from khaosz.config import *
from khaosz.model import *
from khaosz.data import *
from khaosz.inference.generator import EmbeddingEncoderCore, GeneratorCore
from tokenizers import pre_tokenizers

@pytest.fixture
def test_env(request: pytest.FixtureRequest):
    func_name = request.function.__name__
    test_dir = tempfile.mkdtemp(prefix=f"{func_name}_")
    config_path = os.path.join(test_dir, "config.json")
    tokenizer_path = os.path.join(test_dir, "tokenizer.json")
    model_path = os.path.join(test_dir, "model.safetensors")
    
    config = {
        "vocab_size": 1000,
        "dim": 128,
        "n_heads": 4,
        "n_kv_heads": 2,
        "dim_ffn": 256,
        "max_len": 64,
        "n_layers": 2,
        "norm_eps": 1e-5
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    tokenizer = BpeTokenizer()
    sp_token_iter = iter(pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(sp_token_iter, config["vocab_size"], 1)
    tokenizer.save(tokenizer_path)
    
    transformer_config = ModelConfig().load(config_path)
    model = Transformer(transformer_config)
    st.save_file(model.state_dict(), model_path)
    
    yield {
        "test_dir": test_dir,
        "model": model,
        "tokenizer": tokenizer,
        "transformer_config": transformer_config,
    }

    shutil.rmtree(test_dir)

def test_model_parameter(test_env):
    save_dir = os.path.join(test_env["test_dir"], "save")
    model_param = ModelParameter(test_env["model"],test_env["tokenizer"] , test_env["transformer_config"])
    model_param.save(save_dir)
    
    assert os.path.exists(os.path.join(save_dir, "model.safetensors"))
    assert os.path.exists(os.path.join(save_dir, "tokenizer.json"))
    assert os.path.exists(os.path.join(save_dir, "config.json"))

# transformer
def test_transformer(test_env):
    model = test_env["model"]
    input_ids = torch.randint(0, test_env["transformer_config"].vocab_size, 
                              (4, test_env["transformer_config"].max_len))
    output_logits = model(input_ids)["logits"]
    target_shape = (4, test_env["transformer_config"].max_len, test_env["transformer_config"].vocab_size)
    assert output_logits.shape == target_shape
    
# generator
def test_embedding_encoder_core(test_env):
    parameter = ModelParameter(
        test_env["model"],
        test_env["tokenizer"],
        test_env["transformer_config"]
    )
    encoder = EmbeddingEncoderCore(parameter)
    
    single_emb = encoder.encode("测试文本")
    assert isinstance(single_emb, torch.Tensor)
    assert single_emb.shape[-1] == test_env["transformer_config"].dim
    

    batch_emb = encoder.encode(["测试1", "测试2"])
    assert isinstance(batch_emb, list)
    assert len(batch_emb) == 2
    
def test_generator_core(test_env):
    parameter = ModelParameter(
        test_env["model"],
        test_env["tokenizer"],
        test_env["transformer_config"]
    )
    generator = GeneratorCore(parameter)
    input_ids = torch.randint(0, test_env["transformer_config"].vocab_size, (4, 10))
    next_token_id, cache_increase = generator.generate_iterator(
        input_ids=input_ids,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        attn_mask=None,
        kv_caches=None,
        start_pos=0
    )
    
    assert next_token_id.shape == (4, 1)
    assert cache_increase == 10
