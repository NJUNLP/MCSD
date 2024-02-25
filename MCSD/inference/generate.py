from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast

from . import strategies


@dataclass
class DecoderOnlyOutput(ModelOutput):
    """
    Base class for outputs of decoder-only generation models using MCSD.
    """

    sequences: torch.LongTensor
    acceptance_count: int = None
    draft_token_count: int = None
    invocation_count: int = None


class Generator:
    def __init__(self) -> None:
        pass

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        raise NotImplementedError


class BaseGenerator:
    def __init__(
        self,
        model,
        eos_token_id: int,
        max_new_tokens: int = 128,
        temp: float = 1,
    ) -> None:
        self.model = model
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.temp = temp

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        past_key_values = None
        invocation_count = 0

        init_input_len = input_ids.size(-1)

        while True:
            if past_key_values is not None:
                pruned_input_ids = input_ids[:, past_key_values[0][0].size(2) :]
            else:
                pruned_input_ids = input_ids

            outputs: CausalLMOutputWithPast = self.model(
                input_ids=pruned_input_ids,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            logits = outputs.logits
            past_key_values = outputs.past_key_values

            batch_num, seq_len, _ = logits.size()

            if self.temp == 0:
                _, ground_tokens = logits.topk(k=1, dim=-1)  # batch x seq_len x 1
            else:
                ground_probs = torch.softmax(
                    logits / self.temp, dim=-1
                )  # batch x seq_len x hidden_dim

                ground_tokens = torch.multinomial(
                    ground_probs.view(batch_num * seq_len, -1), num_samples=1
                )  # batch*seq_len x 1
            ground_tokens = ground_tokens.view(batch_num, seq_len)

            input_ids = torch.cat(
                (input_ids, ground_tokens[:, -1:].to(input_ids)), dim=1
            )

            invocation_count += 1

            if (
                self.eos_token_id in input_ids[0, -1:]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break
        return DecoderOnlyOutput(sequences=input_ids, invocation_count=invocation_count)


class SpeculativeGenerator:
    def __init__(
        self,
        draft_model,
        target_model,
        eos_token_id: int,
        k_config: Tuple[int],
        max_new_tokens: int = 128,
        draft_model_temp: float = 1,
        target_model_temp: float = 1,
        replacement: bool = False,
        speculative_sampling: bool = True,
        tree_attn: bool = True,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.max_new_tokens = max_new_tokens
        self.strategy: strategies.Strategy = None

        if tree_attn:
            self.strategy = strategies.TreeStrategy(
                draft_model=draft_model,
                target_model=target_model,
                k_config=k_config,
                draft_model_temp=draft_model_temp,
                target_model_temp=target_model_temp,
                replacement=replacement,
                speculative_sampling=speculative_sampling,
            )
        else:
            self.strategy = strategies.BatchStrategy(
                draft_model=draft_model,
                target_model=target_model,
                k_config=k_config,
                draft_model_temp=draft_model_temp,
                target_model_temp=target_model_temp,
                replacement=replacement,
                speculative_sampling=speculative_sampling,
            )

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
    ) -> DecoderOnlyOutput:
        target_model_past_key_values = None
        draft_model_past_key_values = None

        invocation_count = 0
        acceptance_count = 0

        init_input_len = input_ids.size(-1)

        while True:
            draft_output = self.strategy.generate_draft(
                input_ids,
                past_key_values=draft_model_past_key_values,
            )

            draft_model_past_key_values = draft_output.past_key_values

            verification_output = self.strategy.verify(
                input_ids=draft_output.sequences,
                target_model_past_key_values=target_model_past_key_values,
                draft_model_past_key_values=draft_output.past_key_values,
                cand_probs=draft_output.cand_probs,
            )

            input_ids = verification_output.sequences

            draft_model_past_key_values = (
                verification_output.draft_model_past_key_values
            )
            target_model_past_key_values = (
                verification_output.target_model_past_key_values
            )

            invocation_count += 1
            acceptance_count += verification_output.acceptance_count

            if (
                self.eos_token_id in input_ids[0, -self.strategy.max_draft_len :]
                or input_ids.size(-1) - init_input_len >= self.max_new_tokens
            ):
                break
        return DecoderOnlyOutput(
            sequences=input_ids,
            acceptance_count=acceptance_count,
            draft_token_count=invocation_count * self.strategy.max_draft_len,
            invocation_count=invocation_count,
        )
