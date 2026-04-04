import os

import torch

from astrai.config.param_config import ModelParameter
from astrai.inference.generator import EmbeddingEncoderCore, GeneratorCore


def test_model_parameter(test_env):
    save_dir = os.path.join(test_env["test_dir"], "save")
    model_param = ModelParameter(
        test_env["model"], test_env["tokenizer"], test_env["transformer_config"]
    )
    ModelParameter.save(model_param, save_dir)

    assert os.path.exists(os.path.join(save_dir, "model.safetensors"))
    assert os.path.exists(os.path.join(save_dir, "tokenizer.json"))
    assert os.path.exists(os.path.join(save_dir, "config.json"))


# transformer
def test_transformer(test_env):
    model = test_env["model"]
    input_ids = torch.randint(
        0,
        test_env["transformer_config"].vocab_size,
        (4, test_env["transformer_config"].max_len),
    )
    output_logits = model(input_ids)["logits"]
    target_shape = (
        4,
        test_env["transformer_config"].max_len,
        test_env["transformer_config"].vocab_size,
    )
    assert output_logits.shape == target_shape


# generator
def test_embedding_encoder_core(test_env):
    parameter = ModelParameter(
        test_env["model"], test_env["tokenizer"], test_env["transformer_config"]
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
        test_env["model"], test_env["tokenizer"], test_env["transformer_config"]
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
        start_pos=0,
    )

    assert next_token_id.shape == (4, 1)
    assert cache_increase == 10
