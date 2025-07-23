import os
import sys

parent_dir = os.path.join(os.path.dirname(__file__), '..')
abs_parent_dir = os.path.abspath(parent_dir)
sys.path.insert(0, abs_parent_dir)

import tempfile
import json
import shutil
import pytest
from khaosz.module import ParameterLoader, ModelParameter
from khaosz.module.transformer import TransformerConfig, Transformer
import safetensors.torch as st
import torch

@pytest.fixture
def test_env():
    # 创建临时测试环境
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    
    # 创建测试配置
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
    
    # 初始化模型参数
    transformer_config = TransformerConfig(config_path)
    model = Transformer(transformer_config)
    
    yield {
        "test_dir": test_dir,
        "config_path": config_path,
        "transformer_config": transformer_config,
        "model": model
    }
    
    # 清理临时文件
    shutil.rmtree(test_dir)

def test_parameter_loader(test_env):
    # 测试参数加载器
    model_path = os.path.join(test_env["test_dir"], "model.safetensors")
    model = test_env["model"]
    st.save_file(model.state_dict(), model_path)
    
    loaded_param = ParameterLoader.load(test_env["test_dir"])
    assert loaded_param.model is not None

def test_model_parameter(test_env):
    # 测试模型参数类
    save_dir = os.path.join(test_env["test_dir"], "save")
    model_param = ModelParameter(test_env["model"], None, test_env["transformer_config"])
    model_param.save(save_dir)
    
    # 检查保存的文件
    assert os.path.exists(os.path.join(save_dir, "model.safetensors"))
    assert os.path.exists(os.path.join(save_dir, "config.json"))