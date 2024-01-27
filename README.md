# Multi-Candidate Speculative Decoding

## Code Release
See [here](./MCSD/).

## Data Release
For [Alpaca dataset](https://github.com/flexflow/FlexFlow/tree/inference?tab=readme-ov-file#prompt-datasets), we use exactly the same exact source as [SpecInfer](https://arxiv.org/pdf/2305.09781.pdf).

For the [WMT dataset](/dataset/wmt_ende.json), we follow the process of SpecInfer: randomly sampling 1000 samples from the test set. We wrap the source sentences using the following template:
```
Translate the input English sentence into German.
Input: {source sentence}
Output: 
```

## Model Release
We release our fine-tuned draft models on hugginface, see [Vicuna-68M](https://huggingface.co/double7/vicuna-68m) and [Vicuna-160M](https://huggingface.co/double7/vicuna-160m). They are fine-tuned from [LLaMA-68M](https://huggingface.co/JackFram/llama-68m) and [LLaMA-160M](https://huggingface.co/JackFram/llama-160m) respectively on ShareGPT data. The training setup follows [FastChat](https://github.com/lm-sys/FastChat).
