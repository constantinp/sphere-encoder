import torch
from torch import nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None


class QwenTextEmbedder(nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3.5-0.8B-Base",
        extraction_layers=None,
        max_length=512,
        dtype=torch.float16,
        device="cpu",
        cache_dir=None,
        local_files_only=False,
        template="{}",
    ):
        super().__init__()
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("QwenTextEmbedder requires the 'transformers' package.")

        self.model_name = model_name
        self.extraction_layers = list(extraction_layers or [24])
        self.max_length = max_length
        self.dtype = dtype
        self.device = torch.device(device)
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.template = template

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if getattr(self.tokenizer, "pad_token", None) is None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
            if eos_token is None:
                raise ValueError("Tokenizer must define either pad_token or eos_token.")
            self.tokenizer.pad_token = eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self._validate_extraction_layers()
        self.output_dim = len(self.extraction_layers) * self.model.config.hidden_size

    def _validate_extraction_layers(self):
        num_hidden_states = self.model.config.num_hidden_layers + 1
        invalid_layers = [layer for layer in self.extraction_layers if layer < 0 or layer >= num_hidden_states]
        if invalid_layers:
            raise ValueError(
                f"Invalid extraction_layers={invalid_layers}; valid hidden_state indices are [0, {num_hidden_states - 1}]"
            )

    def _format_prompts(self, prompts):
        return [self.template.format(prompt or "") for prompt in prompts]

    def tokenize(self, prompts):
        if not prompts:
            raise ValueError("prompts must not be empty")
        formatted = self._format_prompts(prompts)
        return self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    @torch.no_grad()
    def encode_tokenized(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        if outputs.hidden_states is None:
            raise RuntimeError("Qwen model did not return hidden_states.")
        extracted = [outputs.hidden_states[layer] for layer in self.extraction_layers]
        return torch.cat(extracted, dim=-1)

    @torch.no_grad()
    def encode_with_attention_mask(self, prompts):
        inputs = self.tokenize(prompts)
        hidden_states = self.encode_tokenized(inputs)
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return hidden_states * mask, attention_mask

    @torch.no_grad()
    def pool_hidden_states(self, hidden_states, attention_mask):
        if hidden_states.ndim != 3:
            raise ValueError(f"hidden_states must have rank 3, got {hidden_states.shape}")
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must have rank 2, got {attention_mask.shape}")
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    @torch.no_grad()
    def encode_pooled(self, prompts):
        hidden_states, attention_mask = self.encode_with_attention_mask(prompts)
        return self.pool_hidden_states(hidden_states, attention_mask)

    def forward(self, prompts):
        return self.encode_pooled(prompts)
