import os
import json
import torch
import pytest
import tempfile
import safetensors.torch as st
from khaosz.model.transformer import Transformer
from khaosz.config.model_config import ModelConfig


@pytest.fixture
def transformer_test_env():
    """创建Transformer测试专用环境"""
    test_dir = tempfile.mkdtemp(prefix="transformer_test_")
    config_path = os.path.join(test_dir, "config.json")
    
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
    
    yield {
        "test_dir": test_dir,
        "config_path": config_path,
        "config": config
    }
    
    if os.path.exists(test_dir):
        try:
            for file in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, file))
            os.rmdir(test_dir)
        except:
            pass


def test_tie_weight_init(transformer_test_env):
    config_path = transformer_test_env["config_path"]
    config_data = transformer_test_env["config"].copy()
    
    # case 1: tie weight
    config_data["tie_weight"] = True
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    config = ModelConfig().load(config_path)
    model = Transformer(config)
    
    assert torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    
    original_weight = model.embed_tokens.weight.clone()
    model.embed_tokens.weight.data[0, 0] = 100.0
    
    assert torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert not torch.equal(model.lm_head.weight, original_weight)
    
    # case 2: not tie weight
    config_data["tie_weight"] = False
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    config = ModelConfig().load(config_path)
    model = Transformer(config)
    
    assert not torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()


def test_model_save_load_with_tie_weight(transformer_test_env):
    test_dir = transformer_test_env["test_dir"]
    model_path = os.path.join(test_dir, "model.safetensors")
    
    config_data = transformer_test_env["config"].copy()
    
    # case 1: tie weight
    config_data["tie_weight"] = True
    config_path = os.path.join(test_dir, "config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    config = ModelConfig().load(config_path)
    original_model = Transformer(config)
    
    st.save_file(original_model.state_dict(), model_path)

    loaded_config = ModelConfig().load(config_path)
    model = Transformer(loaded_config)
    model.load_state_dict(st.load_file(model_path))
    
    assert torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    assert "lm_head.weight" not in model.state_dict()

    # case 2: not tie weight
    config_data["tie_weight"] = False
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    config = ModelConfig().load(config_path)
    original_model = Transformer(config)

    st.save_file(original_model.state_dict(), model_path)
    
    loaded_config = ModelConfig().load(config_path)
    model = Transformer(loaded_config)
    model.load_state_dict(st.load_file(model_path))

    assert torch.equal(model.lm_head.weight, model.embed_tokens.weight)
    assert model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr()
    assert "lm_head.weight" in model.state_dict()
    