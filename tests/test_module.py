import os
import sys

parent_dir = os.path.join(os.path.dirname(__file__), '..')
abs_parent_dir = os.path.abspath(parent_dir)
sys.path.insert(0, abs_parent_dir)

import tempfile
import json
import shutil
import pytest
from khaosz.module import ParameterLoader, ModelParameter, BpeTokenizer
from khaosz.module.transformer import TransformerConfig, Transformer
import safetensors.torch as st

@pytest.fixture
def test_env():
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    tokenizer_path = os.path.join(test_dir, "tokenizer.json")
    tokenizer = BpeTokenizer()
    tokenizer.save(tokenizer_path)
    
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
    
    transformer_config = TransformerConfig(config_path)
    model = Transformer(transformer_config)
    
    yield {
        "test_dir": test_dir,
        "config_path": config_path,
        "model": model,
        "tokenizer": tokenizer,
        "transformer_config": transformer_config,
        
    }

    shutil.rmtree(test_dir)

def test_parameter_loader(test_env):
    model_path = os.path.join(test_env["test_dir"], "model.safetensors")
    model = test_env["model"]
    st.save_file(model.state_dict(), model_path)
    
    loaded_param = ParameterLoader.load(test_env["test_dir"])
    assert loaded_param.model is not None

def test_model_parameter(test_env):
    save_dir = os.path.join(test_env["test_dir"], "save")
    model_param = ModelParameter(test_env["model"],test_env["tokenizer"] , test_env["transformer_config"])
    model_param.save(save_dir)
    
    assert os.path.exists(os.path.join(save_dir, "model.safetensors"))
    assert os.path.exists(os.path.join(save_dir, "config.json"))