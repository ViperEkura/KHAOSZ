import os

import torch

from astrai.config.param_config import ModelParameter

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