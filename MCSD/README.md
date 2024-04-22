# Source Code for Multi-Candidate Speculative Decoding

We provide Python application interfaces for inference, as well as command-line interfaces for evaluation.

## Dependencies

PyTorch version >= 1.11.0

Python version >= 3.8

transformers >= 4.34.0

## Evaluation CLI
Run the following script for evaluation:
```
python evaluation.py \
--draft-model PATH_TO_DRAFT_MODEL \
--target-model PATH_TO_TARGET_MODEL \
--fp16 \
--k-config 4,2,2 \
--datapath PATH_TO_DATA \
--sampling-type sampling
```

### Options
```
-h, --help                  show this help message and exit
--draft-model               Draft model path.
--target-model              Target model path.
--tokenizer                 Tokenizer path. If not provided, use the Target model path.
--fp16                      Use float16 dtype.
--k-config                  Use comma separations, e.g. `--k-config 4,2,2`.
--datapath                  The json data file.
--max-new-tokens
--replacement               Sampling with replacement.
--naive-sampling            Use multi-candidate naive sampling.
--disable-tree-attn
--sampling-type             {argmax,sampling}
--disable-tqdm
--auto-model                Use AutoModelForCausalLM and AutoTokenizer to load the model and tokenizer, this will disable the tree attn.
```

Note:
* Tree Attn is currently not supported for models other than LLaMA. Therefore, when using '--auto-model', Tree Attn will be disabled.
* Since flash-attn does not support custom attention masks, it is currently incompatible with Tree Attn.

## Python application interfaces
Here is an example of inference using our generator, see here for the function of each argument.
```python
import torch
from model.llama_tree_attn import LlamaForCausalLM, LlamaTokenizer
from inference.generate import SpeculativeGenerator

draft_model = LlamaForCausalLM.from_pretrained(
    "PATH_TO_DRAFT_MODEL",
    torch_dtype=torch.float16,
    device_map=0,
)
target_model = LlamaForCausalLM.from_pretrained(
    "PATH_TO_TARGET_MODEL",
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = LlamaTokenizer.from_pretrained("PATH_TO_TARGET_MODEL")

generator = SpeculativeGenerator(
    draft_model,
    target_model,
    eos_token_id=tokenizer.eos_token_id,
    k_config=(4, 2, 2),
    max_new_tokens=128,
    draft_model_temp=1,
    target_model_temp=1,
    replacement=False,
    speculative_sampling=True,
    tree_attn=True,
)

prompt_text = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
input_ids = inputs.input_ids
with torch.no_grad():
    output = generator.generate(input_ids)
output_text = tokenizer.batch_decode(
    output.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("Output:\n{}".format(output_text))

```
