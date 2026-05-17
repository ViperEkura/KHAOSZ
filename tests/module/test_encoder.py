import torch

from astrai.config.model_config import EncoderConfig
from astrai.model.encoder import EmbeddingEncoder

TINY_CONFIG = dict(
    vocab_size=128,
    dim=8,
    n_heads=2,
    n_kv_heads=1,
    dim_ffn=16,
    max_len=64,
    n_layers=2,
    norm_eps=1e-5,
)


def test_encoder_forward_mean():
    config = EncoderConfig(**TINY_CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingEncoder(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        output = model(input_ids)

    assert output.shape == (batch_size, config.dim)
    assert not torch.isnan(output).any()


def test_encoder_forward_cls():
    config = EncoderConfig(**{**TINY_CONFIG, "pooling_type": "cls"})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingEncoder(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        output = model(input_ids)

    assert output.shape == (batch_size, config.dim)
    assert not torch.isnan(output).any()


def test_encoder_forward_last():
    config = EncoderConfig(**{**TINY_CONFIG, "pooling_type": "last"})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingEncoder(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        output = model(input_ids)

    assert output.shape == (batch_size, config.dim)
    assert not torch.isnan(output).any()


def test_encoder_forward_with_padding():
    config = EncoderConfig(**TINY_CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingEncoder(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )
    input_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    input_mask[:, 4:] = False

    with torch.no_grad():
        output = model(input_ids, input_mask=input_mask)

    assert output.shape == (batch_size, config.dim)
    assert not torch.isnan(output).any()


def test_encoder_normalize():
    config = EncoderConfig(
        **{**TINY_CONFIG, "pooling_type": "mean", "normalize_embeddings": True}
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingEncoder(config).to(device=device)
    model.eval()

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(
        0, config.vocab_size, (batch_size, seq_len), device=device
    )

    with torch.no_grad():
        output = model(input_ids)

    norms = output.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_encoder_register():
    from astrai.model.automodel import AutoModel

    assert AutoModel.is_registered("embedding")
    cls = AutoModel.get_component_class("embedding")
    assert cls is EmbeddingEncoder


def test_encoder_from_transformer_checkpoint():
    config = EncoderConfig(**TINY_CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EmbeddingEncoder(config).to(device=device)

    state_dict = model.state_dict()
    state_dict["lm_head.weight"] = torch.randn(
        config.vocab_size, config.dim, device=device
    )

    new_model = EmbeddingEncoder(config).to(device=device)
    new_model.load_state_dict(state_dict, strict=True)

    for key in model.state_dict():
        assert torch.equal(new_model.state_dict()[key], model.state_dict()[key])


def test_encoder_save_load():
    import json
    import os
    import tempfile

    import safetensors.torch as st

    test_dir = tempfile.mkdtemp(prefix="encoder_test_")
    config_path = os.path.join(test_dir, "config.json")
    weights_path = os.path.join(test_dir, "model.safetensors")

    try:
        config_data = {**TINY_CONFIG, "pooling_type": "mean"}
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = EncoderConfig.from_file(config_path)
        original = EmbeddingEncoder(config)
        st.save_file(original.state_dict(), weights_path)

        loaded = EmbeddingEncoder(config)
        loaded.load_state_dict(st.load_file(weights_path))

        for key in original.state_dict():
            assert torch.equal(original.state_dict()[key], loaded.state_dict()[key])
    finally:
        if os.path.exists(test_dir):
            for f in os.listdir(test_dir):
                os.remove(os.path.join(test_dir, f))
            os.rmdir(test_dir)
