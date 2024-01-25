import argparse
import json
import logging
import time
from typing import Literal, Tuple

import torch
from inference.generate import Generator
from model.llama_tree_attn import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class JsonData:
    def __init__(self, path) -> None:
        with open(path) as fin:
            self.data = json.load(fin)

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.data[index]

    def __len__(self):
        return len(self.data)


def run_eval(
    draft_model,
    target_model,
    tokenizer,
    k_config: Tuple[int],
    datapath: str,
    max_new_tokens: int = 128,
    replacement=False,
    speculative_sampling=True,
    tree_attn=True,
    sampling_type: Literal["argmax", "sampling"] = "sampling",
    disable_tqdm: bool = False,
):
    if sampling_type not in ["argmax", "sampling"]:
        raise ValueError(
            f'`sampling_type` can be either `"argmax"` or `"sampling"`, but received "{sampling_type}"'
        )
    if sampling_type == "argmax":
        target_model_temp = 0
        draft_model_temp = 0
    else:
        target_model_temp = 1
        draft_model_temp = 1

    dataloader = JsonData(datapath)
    generator = Generator(
        draft_model,
        target_model,
        eos_token_id=tokenizer.eos_token_id,
        k_config=k_config,
        max_new_tokens=max_new_tokens,
        draft_model_temp=draft_model_temp,
        target_model_temp=target_model_temp,
        replacement=replacement,
        speculative_sampling=speculative_sampling,
        tree_attn=tree_attn,
    )

    draft_model.eval()
    target_model.eval()

    logger.info("evaluation start.")
    start_time = time.time()

    acceptance_count = 0
    draft_token_count = 0
    invocation_count = 0

    iterator = range(len(dataloader))
    with torch.no_grad():
        for sample_idx in iterator if disable_tqdm else tqdm(iterator):
            prompt_text = dataloader[sample_idx]
            inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            output = generator.generate(input_ids)

            acceptance_count += output.acceptance_count
            draft_token_count += output.draft_token_count
            invocation_count += output.invocation_count
    end_time = time.time()

    logger.info("evaluation complete.")

    run_time = end_time - start_time

    latency = run_time / (acceptance_count + invocation_count)
    acceptance_rate = acceptance_count / draft_token_count
    block_efficiency = 1 + acceptance_count / invocation_count

    logger.info("Running time: {:.2f} s".format(run_time))
    logger.info("Token latency: {:.2f} ms".format(latency * 1000))
    logger.info("Acceptance rate: {:.2f}".format(acceptance_rate))
    logger.info("Block efficiency: {:.2f}".format(block_efficiency))


def main(args):
    torch_dtype = torch.float16 if args.fp16 else torch.float32

    logger.info("Loading draft model: {}".format(args.draft_model))
    draft_model = LlamaForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch_dtype,
        device_map=0,
    )

    logger.info("Loading target model: {}".format(args.target_model))
    target_model = LlamaForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer)

    run_eval(
        draft_model,
        target_model,
        tokenizer=tokenizer,
        k_config=args.k_config,
        datapath=args.datapath,
        max_new_tokens=args.max_new_tokens,
        replacement=not args.replacement,
        speculative_sampling=not args.naive_sampling,
        tree_attn=not args.disable_tree_attn,
        sampling_type=args.sampling_type,
        disable_tqdm=args.disable_tqdm,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-model", type=str, required=True, help="Draft model path."
    )
    parser.add_argument(
        "--target-model", type=str, required=True, help="Target model path."
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path.")
    parser.add_argument("--fp16", action="store_true", help="use float16 dtype.")

    parser.add_argument(
        "--k-config",
        type=lambda x: tuple(map(int, x.split(","))),
        required=True,
        help="Use comma separations, e.g. `--k-config 4,2,2`.",
    )

    parser.add_argument(
        "--datapath", type=str, required=True, help="The json data file."
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--replacement",
        action="store_true",
        help="Sampling with replacement.",
    )
    parser.add_argument(
        "--naive-sampling",
        action="store_true",
        help="Use multi-candidate naive sampling.",
    )

    parser.add_argument("--disable-tree-attn", action="store_true")

    parser.add_argument(
        "--sampling-type", type=str, default="sampling", choices=["argmax", "sampling"]
    )

    parser.add_argument("--disable_tqdm", action="store_true")

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.target_model
    main(args)
