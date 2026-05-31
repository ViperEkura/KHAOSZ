import json
import os
import tempfile

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from astrai.config.preprocess_config import (
    InputConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from astrai.preprocessing.builder import (
    MaskBuilderFactory,
    SectionedMaskBuilder,
)
from astrai.preprocessing.pipeline import Pipeline, dedup_signature, filter_by_length
from astrai.tokenize import AutoTokenizer

_SPECIAL_TOKENS_CONFIG = {
    "bos_token": "<|begin_of_sentence|>",
    "eos_token": "<|end_of_sentence|>",
    "pad_token": "<|_pad_|>",
    "unk_token": "<|_unk_|>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
}

_SPECIAL_TOKENS = list(_SPECIAL_TOKENS_CONFIG.values())

_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _build_chat_tokenizer() -> AutoTokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(
        vocab_size=512,
        min_frequency=1,
        special_tokens=_SPECIAL_TOKENS,
    )
    train_data = [
        "hello world",
        "Hi there!",
        "You are helpful.",
        "What is 2+2?",
        "Tell me a story about dragons and knights.",
        "Sure, here is a tale.",
        "Translate to French: Hello",
        "Bonjour",
        "Artificial Intelligence is a field of computer science.",
        "system",
        "user",
        "assistant",
        "<|im_start|>",
        "<|im_end|>",
        *[chr(i) for i in range(32, 127)],
    ]
    tok.train_from_iterator(train_data, tr)

    auto_tok = AutoTokenizer()
    auto_tok._tokenizer = tok
    auto_tok._special_token_map = {
        "bos_token": "<|begin_of_sentence|>",
        "eos_token": "<|end_of_sentence|>",
        "pad_token": "<|_pad_|>",
        "unk_token": "<|_unk_|>",
    }
    auto_tok.set_chat_template(_CHAT_TEMPLATE)
    return auto_tok


@pytest.fixture(scope="session")
def chat_tokenizer():
    return _build_chat_tokenizer()


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors=True)


_CHAT_SECTIONS = [{"field": "messages", "action": "$role", "template": True}]

_INSTRUCTION_SECTIONS = [
    {"field": "prompt", "action": "mask", "add_special_tokens": True},
    {"field": "response", "action": "train"},
]

_TEXT_SECTIONS = [{"field": "text", "action": "train"}]


def make_chat_config():
    return PipelineConfig(
        input=InputConfig(sections=_CHAT_SECTIONS),
        mask={"system": "mask", "user": "mask", "assistant": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )


def make_instruction_config():
    return PipelineConfig(
        input=InputConfig(sections=_INSTRUCTION_SECTIONS),
        mask={"prompt": "mask", "response": "train"},
        mask_default="mask",
        preprocessing=ProcessingConfig(max_seq_len=2048),
    )


def make_text_config():
    return PipelineConfig(
        input=InputConfig(sections=_TEXT_SECTIONS),
        preprocessing=ProcessingConfig(
            max_seq_len=2048, min_chars=1, max_chars=2_000_000
        ),
    )


class TestPipelineConfig:
    def test_default_values(self):
        config = PipelineConfig()
        assert config.version == 1
        assert config.mask == {}
        assert config.mask_default == "mask"
        assert config.preprocessing.max_seq_len == 2048
        assert config.output.storage_format == "bin"
        assert config.input.sections is None

    def test_from_dict_flat(self):
        data = {
            "version": 1,
            "input": {
                "sections": [{"field": "messages", "action": "$role", "template": True}]
            },
            "mask": {"system": "mask", "assistant": "train"},
            "mask_default": "mask",
            "preprocessing": {"max_seq_len": 1024},
            "output": {"storage_format": "h5"},
        }
        config = PipelineConfig.from_dict(data)
        assert config.input.sections == [
            {"field": "messages", "action": "$role", "template": True}
        ]
        assert config.mask == {"system": "mask", "assistant": "train"}
        assert config.preprocessing.max_seq_len == 1024
        assert config.output.storage_format == "h5"

    def test_to_dict_roundtrip(self):
        config = PipelineConfig(
            input=InputConfig(sections=_INSTRUCTION_SECTIONS),
            mask={"prompt": "mask", "response": "train"},
            mask_default="mask",
        )
        d = config.to_dict()
        config2 = PipelineConfig.from_dict(d)
        assert config2.input.sections == _INSTRUCTION_SECTIONS
        assert config2.mask == {"prompt": "mask", "response": "train"}

    def test_to_json_from_json(self, temp_dir):
        config = PipelineConfig(
            input=InputConfig(sections=_TEXT_SECTIONS),
            mask={"text": "train"},
            mask_default="mask",
        )
        path = os.path.join(temp_dir, "config.json")
        config.to_json(path)
        loaded = PipelineConfig.from_json(path)
        assert loaded.input.sections == _TEXT_SECTIONS
        assert loaded.mask == {"text": "train"}


class TestChatMaskBuilder:
    def test_simple_chat_mask(self, chat_tokenizer):
        config = make_chat_config()
        builder = SectionedMaskBuilder()
        item = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = builder.build(item, config, chat_tokenizer)
        assert result is not None
        assert "ids" in result
        assert "loss_mask" in result
        assert len(result["ids"]) == len(result["loss_mask"])

        ids = chat_tokenizer.decode(result["ids"], skip_special_tokens=False)

        assert "system" in ids.lower() or "<|im_start|>system" in ids
        assert "assistant" in ids.lower() or "<|im_start|>assistant" in ids

        total = len(result["ids"])
        trained = sum(result["loss_mask"])
        assert trained > 0, "At least assistant tokens should be trained"
        assert trained < total, "System and user tokens should be masked"

    def test_mask_only_assistant_trained(self, chat_tokenizer):
        config = make_chat_config()
        builder = SectionedMaskBuilder()
        item = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        result = builder.build(item, config, chat_tokenizer)
        mask = result["loss_mask"]
        ids = result["ids"]

        assert len(ids) == len(mask)

        trained_positions = [i for i, m in enumerate(mask) if m == 1]
        assert len(trained_positions) > 0, "At least some tokens should be trained"

        masked_positions = [i for i, m in enumerate(mask) if m == 0]
        assert len(masked_positions) > 0, "User tokens should be masked"

    def test_chat_all_masked(self, chat_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_CHAT_SECTIONS),
            mask={"system": "mask", "user": "mask", "assistant": "mask"},
            mask_default="mask",
            preprocessing=ProcessingConfig(max_seq_len=2048),
        )
        builder = SectionedMaskBuilder()
        item = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = builder.build(item, config, chat_tokenizer)
        assert sum(result["loss_mask"]) == 0

    def test_chat_all_trained(self, chat_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_CHAT_SECTIONS),
            mask={},
            mask_default="train",
            preprocessing=ProcessingConfig(max_seq_len=2048),
        )
        builder = SectionedMaskBuilder()
        item = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        result = builder.build(item, config, chat_tokenizer)
        assert sum(result["loss_mask"]) == len(result["ids"]) - 1

    def test_empty_messages_returns_none(self, chat_tokenizer):
        config = make_chat_config()
        builder = SectionedMaskBuilder()
        assert builder.build({"messages": []}, config, chat_tokenizer) is None
        assert builder.build({}, config, chat_tokenizer) is None

    def test_domain_extraction(self, chat_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_CHAT_SECTIONS),
            mask={"assistant": "train"},
            mask_default="mask",
            preprocessing=ProcessingConfig(max_seq_len=2048),
            output=OutputConfig(domain_key="source"),
        )
        builder = SectionedMaskBuilder()
        item = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
            "source": "wiki",
        }
        result = builder.build(item, config, chat_tokenizer)
        assert result["domain"] == "wiki"

    def test_truncation_to_max_len(self, chat_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_CHAT_SECTIONS),
            mask={"assistant": "train"},
            mask_default="mask",
            preprocessing=ProcessingConfig(max_seq_len=10),
        )
        builder = SectionedMaskBuilder()
        item = {
            "messages": [
                {
                    "role": "user",
                    "content": "Tell me a very long story about dragons and knights and magic.",
                },
                {"role": "assistant", "content": "Sure! Here is a tale..."},
            ]
        }
        result = builder.build(item, config, chat_tokenizer)
        assert len(result["ids"]) <= 10
        assert len(result["loss_mask"]) == len(result["ids"])


class TestInstructionMaskBuilder:
    def test_basic_instruction_mask(self, test_tokenizer):
        config = make_instruction_config()
        builder = SectionedMaskBuilder()
        item = {"prompt": "Translate to French: Hello", "response": "Bonjour"}
        result = builder.build(item, config, test_tokenizer)
        assert result is not None
        assert len(result["ids"]) == len(result["loss_mask"])

    def test_prompt_masked_response_trained(self, test_tokenizer):
        config = make_instruction_config()
        builder = SectionedMaskBuilder()
        item = {"prompt": "hello", "response": "world"}
        result = builder.build(item, config, test_tokenizer)
        mask = result["loss_mask"]
        ids = result["ids"]

        prompt_ids = test_tokenizer.encode("hello", add_special_tokens=True)
        response_ids = test_tokenizer.encode("world", add_special_tokens=False)

        p_len = min(len(prompt_ids), len(ids))
        assert all(m == 0 for m in mask[:p_len])

        if p_len < len(ids):
            assert all(m == 1 for m in mask[p_len:])

    def test_train_on_prompt(self, test_tokenizer):
        config = PipelineConfig(
            input=InputConfig(
                sections=[
                    {
                        "field": "prompt",
                        "action": "train",
                        "add_special_tokens": True,
                    },
                    {"field": "response", "action": "mask"},
                ]
            ),
            preprocessing=ProcessingConfig(max_seq_len=2048),
        )
        builder = SectionedMaskBuilder()
        item = {"prompt": "hello", "response": "world"}
        result = builder.build(item, config, test_tokenizer)
        mask = result["loss_mask"]
        ids = result["ids"]

        prompt_ids = test_tokenizer.encode("hello", add_special_tokens=True)
        p_len = min(len(prompt_ids), len(ids))
        assert all(m == 1 for m in mask[:p_len])


class TestTextMaskBuilder:
    def test_basic_text(self, test_tokenizer):
        config = make_text_config()
        builder = SectionedMaskBuilder()
        item = {"text": "Hello world. This is a test document."}
        result = builder.build(item, config, test_tokenizer)
        assert result is not None
        assert "ids" in result
        assert len(result["ids"]) > 0
        assert "loss_mask" not in result

    def test_empty_text_returns_none(self, test_tokenizer):
        config = make_text_config()
        builder = SectionedMaskBuilder()
        assert builder.build({"text": ""}, config, test_tokenizer) is None
        assert builder.build({"text": "   "}, config, test_tokenizer) is None

    def test_too_short_text(self, test_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_TEXT_SECTIONS),
            preprocessing=ProcessingConfig(min_chars=100),
        )
        builder = SectionedMaskBuilder()
        assert builder.build({"text": "short"}, config, test_tokenizer) is None

    def test_truncation(self, test_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_TEXT_SECTIONS),
            preprocessing=ProcessingConfig(max_seq_len=3, min_chars=1),
        )
        builder = SectionedMaskBuilder()
        item = {"text": "This is a very long text that should be truncated"}
        result = builder.build(item, config, test_tokenizer)
        assert len(result["ids"]) <= 3


class TestPipeline:
    def test_full_chat_pipeline(self, temp_dir, chat_tokenizer):
        tokenizer_dir = os.path.join(temp_dir, "tok")
        os.makedirs(tokenizer_dir, exist_ok=True)
        chat_tokenizer._tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
            json.dump(
                {
                    "special_tokens": _SPECIAL_TOKENS_CONFIG,
                    "chat_template": _CHAT_TEMPLATE,
                },
                f,
            )

        jsonl_path = os.path.join(temp_dir, "chat.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "You are helpful."},
                            {"role": "user", "content": "Hi."},
                            {"role": "assistant", "content": "Hello!"},
                        ]
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ]
                    }
                )
                + "\n"
            )

        config = PipelineConfig(
            input=InputConfig(sections=_CHAT_SECTIONS),
            mask={"system": "mask", "user": "mask", "assistant": "train"},
            mask_default="mask",
            preprocessing=ProcessingConfig(max_seq_len=2048, deduplicate=True),
            output=OutputConfig(storage_format="bin", domain_key=None),
        )

        out_dir = os.path.join(temp_dir, "output")
        Pipeline(
            config=config,
            input_paths=[jsonl_path],
            output_dir=out_dir,
            tokenizer_path=tokenizer_dir,
        ).run()

        meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert "sequence" in meta
        assert "loss_mask" in meta
        assert meta["sequence"]["dtype"] == "int32"
        assert meta["loss_mask"]["dtype"] == "int32"

    def test_full_text_pipeline(self, temp_dir, test_tokenizer):

        tokenizer_dir = os.path.join(temp_dir, "tok")
        os.makedirs(tokenizer_dir, exist_ok=True)

        test_tokenizer._tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
            json.dump(
                {
                    "special_tokens": {
                        "pad_token": "<|_pad_|>",
                        "unk_token": "<|_unk_|>",
                    }
                },
                f,
            )

        jsonl_path = os.path.join(temp_dir, "text.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "text": "Hello world this is a test document with enough characters to pass the minimum length filter."
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "text": "Another document for testing purposes with sufficient length to be processed."
                    }
                )
                + "\n"
            )

        config = PipelineConfig(
            input=InputConfig(sections=_TEXT_SECTIONS),
            preprocessing=ProcessingConfig(
                max_seq_len=2048, min_chars=10, deduplicate=True
            ),
            output=OutputConfig(storage_format="bin"),
        )

        out_dir = os.path.join(temp_dir, "output")
        Pipeline(
            config=config,
            input_paths=[jsonl_path],
            output_dir=out_dir,
            tokenizer_path=tokenizer_dir,
        ).run()

        meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert "sequence" in meta
        assert "loss_mask" not in meta
        assert meta["sequence"]["dtype"] == "int32"

    def test_full_instruction_pipeline(self, temp_dir, test_tokenizer):
        tokenizer_dir = os.path.join(temp_dir, "tok")
        os.makedirs(tokenizer_dir, exist_ok=True)
        test_tokenizer._tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
            json.dump(
                {
                    "special_tokens": {
                        "pad_token": "<|_pad_|>",
                        "unk_token": "<|_unk_|>",
                    }
                },
                f,
            )

        jsonl_path = os.path.join(temp_dir, "instruct.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Tell me a joke",
                        "response": "Why did the chicken cross the road?",
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "prompt": "What is AI?",
                        "response": "Artificial Intelligence is a field of computer science.",
                    }
                )
                + "\n"
            )

        config = PipelineConfig(
            input=InputConfig(sections=_INSTRUCTION_SECTIONS),
            mask={"prompt": "mask", "response": "train"},
            mask_default="mask",
            preprocessing=ProcessingConfig(max_seq_len=2048),
            output=OutputConfig(storage_format="bin"),
        )

        out_dir = os.path.join(temp_dir, "output")
        Pipeline(
            config=config,
            input_paths=[jsonl_path],
            output_dir=out_dir,
            tokenizer_path=tokenizer_dir,
        ).run()

        meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
        assert os.path.exists(meta_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert "sequence" in meta
        assert "loss_mask" in meta
        assert meta["sequence"]["dtype"] == "int32"
        assert meta["loss_mask"]["dtype"] == "int32"

    def test_dtype_override(self, temp_dir, test_tokenizer):
        tokenizer_dir = os.path.join(temp_dir, "tok")
        os.makedirs(tokenizer_dir, exist_ok=True)
        test_tokenizer._tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w") as f:
            json.dump(
                {
                    "special_tokens": {
                        "pad_token": "<|_pad_|>",
                        "unk_token": "<|_unk_|>",
                    }
                },
                f,
            )

        jsonl_path = os.path.join(temp_dir, "data.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "prompt": "Q",
                        "response": "A",
                    }
                )
                + "\n"
            )

        config = PipelineConfig(
            input=InputConfig(sections=_INSTRUCTION_SECTIONS),
            mask={"prompt": "mask", "response": "train"},
            mask_default="mask",
            preprocessing=ProcessingConfig(max_seq_len=2048),
            output=OutputConfig(
                storage_format="bin",
                dtype={"loss_mask": "bool"},
            ),
        )

        out_dir = os.path.join(temp_dir, "output")
        Pipeline(
            config=config,
            input_paths=[jsonl_path],
            output_dir=out_dir,
            tokenizer_path=tokenizer_dir,
        ).run()

        meta_path = os.path.join(out_dir, "__default__", "shard_0000", "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        assert meta["sequence"]["dtype"] == "int32"
        assert meta["loss_mask"]["dtype"] == "bool"


class TestUtility:
    def test_filter_by_length(self):
        assert filter_by_length("hello world", min_len=5)
        assert not filter_by_length("hi", min_len=5)
        assert not filter_by_length("x" * 100, max_len=50)
        assert filter_by_length("just right", min_len=5, max_len=20)

    def test_dedup_signature(self):
        a = {"key": "value", "number": 1}
        b = {"number": 1, "key": "value"}
        assert dedup_signature(a) == dedup_signature(b)
        c = {"key": "different"}
        assert dedup_signature(a) != dedup_signature(c)


class TestSectionedMaskBuilder:
    def test_sectioned_chat(self, chat_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_CHAT_SECTIONS),
            mask={"system": "mask", "user": "mask", "assistant": "train"},
            mask_default="mask",
            preprocessing=ProcessingConfig(max_seq_len=2048),
        )
        builder = SectionedMaskBuilder()
        item = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        result = builder.build(item, config, chat_tokenizer)
        assert result is not None
        assert len(result["ids"]) == len(result["loss_mask"])
        assert sum(result["loss_mask"]) > 0
        assert 0 in result["loss_mask"]

    def test_sectioned_instruction(self, test_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_INSTRUCTION_SECTIONS),
            preprocessing=ProcessingConfig(max_seq_len=2048, min_chars=0),
        )
        builder = SectionedMaskBuilder()
        item = {"prompt": "Q: Why?", "response": "A: Because."}
        result = builder.build(item, config, test_tokenizer)
        assert result is not None
        mask = result["loss_mask"]
        assert mask[0] == 0
        assert mask[-1] == 1

    def test_sectioned_text(self, test_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_TEXT_SECTIONS),
            preprocessing=ProcessingConfig(max_seq_len=2048, min_chars=1),
        )
        builder = SectionedMaskBuilder()
        item = {"text": "Hello world, this is a test."}
        result = builder.build(item, config, test_tokenizer)
        assert result is not None
        assert "loss_mask" not in result

    def test_sectioned_text_too_short(self, test_tokenizer):
        config = PipelineConfig(
            input=InputConfig(sections=_TEXT_SECTIONS),
            preprocessing=ProcessingConfig(max_seq_len=2048, min_chars=100),
        )
        builder = SectionedMaskBuilder()
        item = {"text": "short"}
        result = builder.build(item, config, test_tokenizer)
        assert result is None


class TestFactoryRegistration:
    def test_registered_builders(self):
        names = MaskBuilderFactory._registry.list_names()
        assert "sectioned" in names

    def test_create_sectioned_builder(self):
        builder = MaskBuilderFactory.create("sectioned")
        assert isinstance(builder, SectionedMaskBuilder)
