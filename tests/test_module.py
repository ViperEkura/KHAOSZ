import os
import sys



parent_dir = os.path.join(os.path.dirname(__file__), '..')
abs_parent_dir = os.path.abspath(parent_dir)
sys.path.insert(0, abs_parent_dir)

import json
import torch
import shutil
import pytest
import tempfile
import safetensors.torch as st
from khaosz.module import *
from khaosz.module.generator import EmbeddingEncoderCore, GeneratorCore
from tokenizers import pre_tokenizers

@pytest.fixture
def test_env():
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    tokenizer_path = os.path.join(test_dir, "tokenizer.json")
    model_path = os.path.join(test_dir, "model.safetensors")
    
    config = {
        "vocab_size": 1000,
        "n_dim": 128,
        "n_head": 4,
        "n_kvhead": 2,
        "d_ffn": 256,
        "m_len": 64,
        "n_layer": 2,
        "norm_eps": 1e-5
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    tokenizer = BpeTokenizer()
    sp_token_iter = iter(pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator(sp_token_iter, config["vocab_size"], 1)
    tokenizer.save(tokenizer_path)
    
    transformer_config = TransformerConfig(config_path)
    model = Transformer(transformer_config)
    st.save_file(model.state_dict(), model_path)
    
    yield {
        "test_dir": test_dir,
        "model": model,
        "tokenizer": tokenizer,
        "transformer_config": transformer_config,
    }

    shutil.rmtree(test_dir)

# parameter loader
def test_parameter_loader(test_env):
    loaded_param = ParameterLoader.load(test_env["test_dir"])
    assert loaded_param.model is not None
    assert loaded_param.tokenizer is not None
    assert loaded_param.config == test_env["transformer_config"]

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
                              (4, test_env["transformer_config"].m_len))
    output_logits = model(input_ids)
    target_shape = (4, test_env["transformer_config"].m_len, test_env["transformer_config"].vocab_size)
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
    assert single_emb.shape[-1] == test_env["transformer_config"].n_dim
    

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
    
    next_token = generator.sample_next_token(
        ids=[1, 2, 3],
        temperature=0.5,
        top_k=10,
        top_p=0.9
    )
    assert isinstance(next_token, int)
    
    next_tokens = generator.sample_next_token(
        ids=[[1, 2, 3], [4, 5, 6]],
        temperature=0.5,
        top_k=10,
        top_p=0.9
    )
    assert isinstance(next_tokens, list)
    
    

