import gc
import re
from pathlib import Path

import pandas as pd


MODEL_PATH = ""  # Local ProstT5 model directory.
INPUT_PATH = ""  # Input Excel or CSV file.
OUTPUT_PATH = ""  # Output Excel or CSV file.
BATCH_SIZE = 32
DEVICE = "cuda:0"

THREEDI_ALPHABET = "acdefghiklmnpqrstvwy"
EXTRA_PATTERN = re.compile(r"<extra_(\d+)>")
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(value):
    sequence = str(value).upper().strip()
    sequence = re.sub(r"[\s*]", "", sequence)
    return "".join(aa if aa in VALID_AA else "X" for aa in sequence)


def translate_extra_tokens(raw):
    indices = [int(value) for value in EXTRA_PATTERN.findall(raw)]
    if not indices:
        raise ValueError(
            "No <extra_i> tokens were decoded. Check the original model and "
            "tokenizer; this script does not remap native 3Di letters."
        )
    translated = "".join(
        THREEDI_ALPHABET[index]
        for index in indices if index < len(THREEDI_ALPHABET)
    )
    return translated


def read_input(path):
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Input must be an .xlsx or .csv file.")

    uid_column = next(
        (name for name in ("UniProt ID", "UniProt_ID") if name in frame), None
    )
    if uid_column is None or not {"Position", "Sequence"}.issubset(frame):
        raise ValueError("Required columns: UniProt ID (or UniProt_ID), Position, Sequence.")
    if frame.empty:
        raise ValueError("Input contains no sites.")
    if frame[[uid_column, "Position", "Sequence"]].isna().any().any():
        raise ValueError("Site identifiers or sequences contain missing values.")

    uids = frame[uid_column].astype(str).str.strip()
    positions = pd.to_numeric(frame["Position"], errors="raise")
    if uids.eq("").any() or not ((positions > 0) & (positions % 1 == 0)).all():
        raise ValueError("Accessions must be nonempty and positions positive integers.")
    keys = pd.DataFrame({"uid": uids, "position": positions})
    if keys.duplicated().any():
        raise ValueError("Duplicate accession-position pairs.")

    sequences = frame["Sequence"].map(clean_sequence)
    valid = sequences.str.len().eq(51) & sequences.str[25].eq("K")
    if not valid.all():
        rows = (frame.index[~valid] + 2).tolist()[:10]
        raise ValueError(f"Expected 51-residue windows with K at position 26; spreadsheet rows: {rows}")
    return frame, sequences.tolist()


def load_model(model_path, device):
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )
    except Exception:
        tokenizer = T5Tokenizer.from_pretrained(
            model_path, do_lower_case=False, legacy=True
        )

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = T5ForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    if len(tokenizer) < model.config.vocab_size:
        gap = model.config.vocab_size - len(tokenizer)
        tokenizer.add_tokens([f"<extra_{i}>" for i in range(gap)])
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError("Tokenizer size does not match model vocabulary size.")
    return tokenizer, model.to(device).eval()


def generate_batch(sequences, tokenizer, model, device):
    import torch

    texts = ["<AA2fold> " + " ".join(sequence) for sequence in sequences]
    encoded = tokenizer.batch_encode_plus(
        texts, add_special_tokens=True, padding="longest", return_tensors="pt"
    ).to(device)
    try:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                max_length=max(map(len, sequences)) + 50,
                do_sample=False,
                num_beams=1,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(decoded) != len(sequences):
            raise ValueError("Output count differs from input count.")
        return [translate_extra_tokens(raw.replace(" ", "").lower()) for raw in decoded]
    except torch.cuda.OutOfMemoryError:
        del encoded
        gc.collect()
        torch.cuda.empty_cache()
        if len(sequences) == 1:
            raise RuntimeError("Insufficient GPU memory for one sequence.") from None
        midpoint = len(sequences) // 2
        return (
            generate_batch(sequences[:midpoint], tokenizer, model, device)
            + generate_batch(sequences[midpoint:], tokenizer, model, device)
        )


def main():
    import torch
    from tqdm import tqdm

    if not all((MODEL_PATH.strip(), INPUT_PATH.strip(), OUTPUT_PATH.strip())):
        raise ValueError("Set MODEL_PATH, INPUT_PATH, and OUTPUT_PATH before running.")
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    if BATCH_SIZE <= 0:
        raise ValueError("Batch size must be positive.")
    if output_path.suffix.lower() not in {".xlsx", ".csv"}:
        raise ValueError("Output must be an .xlsx or .csv file.")
    if output_path.exists():
        raise FileExistsError(f"Output already exists; choose another OUTPUT_PATH: {output_path}")

    frame, sequences = read_input(input_path)
    device = torch.device(DEVICE)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the environment or set DEVICE to cpu.")
    tokenizer, model = load_model(MODEL_PATH, device)

    results = []
    for start in tqdm(range(0, len(sequences), BATCH_SIZE), desc="Generating 3Di"):
        results.extend(generate_batch(
            sequences[start:start + BATCH_SIZE], tokenizer, model, device
        ))
    if len(results) != len(frame):
        raise RuntimeError("Output count differs from input count.")

    output = frame.copy()
    output["foldseek_seq"] = results
    output["3Di_Status"] = [
        "valid_51" if len(value) == 51 else "fallback_length"
        for value in results
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".xlsx":
        output.to_excel(output_path, index=False)
    else:
        output.to_csv(output_path, index=False)

    del model, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
