weighted, category-balanced dataset builder for LLM fine-tuning. takes a directory of JSONL conversation files, samples from each category according to configurable weights, and outputs a single shuffled dataset. two scripts, zero dependencies beyond pandas.

```bash
python dataset-chooser.py
python dataset-chooser.py --config my-config.ini   # custom config path
python dataset-evaluator.py --help                  # show all options
```

[![python](https://img.shields.io/badge/python-3-93450a.svg?style=flat-square)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-grey.svg?style=flat-square)](https://opensource.org/licenses/MIT)

---

## what it does

you have a pile of JSONL files with OpenAI chat-format conversations, split across categories. you want a single fine-tuning dataset where each category contributes a controlled proportion. this does that.

- **parallel file loading** — reads all JSONL files concurrently via thread pool
- **weighted sampling** — each category gets `total_examples * weight` rows. oversamples with replacement if source data is smaller than target
- **category extraction** — uses the first assistant message's content as the category label
- **shuffled output** — final dataset is shuffled and written as JSONL
- **evaluation script** — inspects the output, counts category distribution, renders a terminal table

## install

```bash
pip install pandas rich
```

`rich` is only needed for the evaluator's terminal output. `dataset-chooser` only needs `pandas`.

## configure

edit `config.ini`:

```ini
[Paths]
jsonl_directory = /path/to/your/jsonl/files
output_file = /path/to/your/output/dataset.jsonl

[Weights]
category_weights = {
    "Category1": 0.05,
    "Category2": 0.10,
    "Category3": 0.85
}

[Settings]
total_examples = 1000000
```

weights should sum to 1.0 (not enforced, but your math will be off otherwise).

## usage

### build the dataset

```bash
python dataset-chooser.py
python dataset-chooser.py --config path/to/config.ini
```

reads every `.jsonl` file in `jsonl_directory`, extracts categories from assistant messages, samples according to weights, writes the result to `output_file`.

### inspect the result

```bash
python dataset-evaluator.py
python dataset-evaluator.py --config path/to/config.ini
```

reads the output file, counts unique assistant responses, shows a formatted table with counts and percentages.

## input format

each JSONL file should have one JSON object per line, OpenAI chat format:

```jsonl
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "CategoryName"}]}
```

the first `assistant` message's `content` is used as the category label. this must match a key in `category_weights`.

## configuration reference

| section | key | description |
|:---|:---|:---|
| `[Paths]` | `jsonl_directory` | directory to scan for `.jsonl` input files |
| `[Paths]` | `output_file` | path for the output dataset |
| `[Weights]` | `category_weights` | JSON object mapping category names to float weights |
| `[Settings]` | `total_examples` | target total rows in the output |

both scripts read `config.ini` from the current working directory.

## project structure

```
dataset-chooser.py      — builds the weighted dataset  (prog: dataset-chooser)
dataset-evaluator.py    — inspects category distribution in the output  (prog: dataset-evaluator)
config.ini              — paths, weights, target size
```

## license

MIT
