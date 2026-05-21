# Qwen3 SigLIP VLM Data Format

Qwen3 SigLIP VLM training uses a small set of normalized runtime formats. Dataset-specific conversion should happen offline before training. The runtime transform does not dispatch by dataset name.

## Config Fields

Each data source declares its training semantics explicitly:

```yaml
preprocess: plaintext | conversation | interleaved
text_keys: text | conversations | texts
image_keys: null | image | images
domain: pt | vlm_sft | vlm_pt | ...
```

`names` remains the data source identity for logging, sampling statistics, and debugging. It is not used as a preprocessor registry key for this transform.

## `plaintext`

Pure text pretraining. The sample contains a text field and no image placeholders.

```json
{
  "id": "sample_id",
  "text": "plain pretraining text",
  "domain": "pt"
}
```

## `conversation`

Text or multimodal QA/SFT in ShareGPT/LLaVA format. Image placeholders are represented by `<image>` in the human turn. If images are present, the sample uses `image` for one image or `images` for multiple images.

```json
{
  "id": "sample_id",
  "image": {"bytes": "<image-bytes>", "path": null},
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is shown?"},
    {"from": "gpt", "value": "A small image."}
  ],
  "domain": "vlm_sft"
}
```

Text-only SFT uses the same `conversations` field without image placeholders and without image data.

## `interleaved`

Image-text interleaved pretraining uses aligned `images` and `texts` lists. The lists must have equal length, and each position must contain exactly one non-null value.

```json
{
  "id": "sample_id",
  "images": [
    {"bytes": "<image1-bytes>", "path": null},
    null,
    {"bytes": "<image2-bytes>", "path": null},
    null
  ],
  "texts": [
    null,
    "first text span",
    null,
    "second text span"
  ],
  "domain": "vlm_pt"
}
```

A caption sample is a single-image interleaved sample:

```json
{
  "images": [{"bytes": "<image-bytes>", "path": null}, null],
  "texts": [null, "caption text"],
  "domain": "caption"
}
```

The transform validates that the number of image placeholders matches the number of loaded images before image patchification.
