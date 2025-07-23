import os
import sys

parent_dir = os.path.join(os.path.dirname(__file__), '..')
abs_parent_dir = os.path.abspath(parent_dir)
sys.path.insert(0, abs_parent_dir)

import tempfile
import shutil
import pytest
from khaosz.module.tokenizer import BpeTokenizer

@pytest.fixture
def test_env():
    # 创建临时测试环境
    test_dir = tempfile.mkdtemp()
    yield test_dir
    # 清理临时文件
    shutil.rmtree(test_dir)

def test_tokenizer(test_env):
    # 测试分词器基础功能
    tokenizer = BpeTokenizer()
    assert tokenizer is not None
    
    # 测试编码解码
    text = "Hello, world!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text
    
    # 测试特殊token
    assert tokenizer.bos_id > 0
    assert tokenizer.eos_id > 0
    assert tokenizer.pad_id > 0

def test_tokenizer_saving(test_env):
    # 测试分词器保存/加载
    tokenizer = BpeTokenizer()
    save_path = os.path.join(test_env, "tokenizer.json")
    tokenizer.save(save_path)
    
    loaded_tokenizer = BpeTokenizer(save_path)
    assert len(tokenizer) == len(loaded_tokenizer)