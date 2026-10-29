from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from Bio.PDB import DSSP, PDBParser
from tqdm import tqdm


DATA_PATH = ""  # Input Excel file with UniProt ID, Position, and Sequence.
AFDB_PDB_DIR = ""  # Local PDB cache directory.
OUTPUT_NPZ = ""  # Output structural feature NPZ file.
STATUS_CSV = ""  # Output per-site status CSV file.

WINDOW_SIZE = 51
CENTER_INDEX = 25
AFDB_MODEL_VERSION = 4
REQUEST_TIMEOUT_SECONDS = 30


AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "ASX": "B", "GLX": "Z", "UNK": "X",
}
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_ID = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}
AA_UNKNOWN_ID = AA_TO_ID["X"]


DSSP_TO_ID = {
    "H": 0,
    "E": 1,
    "B": 1,
    "C": 2,
    "G": 3,
    "I": 4,
    "T": 5,
    "S": 6,
    "-": 7,
    " ": 7,
}
SS_UNKNOWN_ID = 8


def normalize_uid(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace("U", "X").replace("Z", "X").replace("O", "X")


def sample_key(uid: str, position: int) -> str:
    return f"{uid}|{position}"


def require_mkdssp() -> str:
    executable = shutil.which("mkdssp") or shutil.which("dssp")
    if executable is None:
        raise RuntimeError(
            "DSSP executable was not found. Install DSSP/mkdssp first, for example "
            "with `conda install -c conda-forge dssp`, then rerun this script."
        )
    return executable


def has_atom_record(path: Path) -> bool:


    try:
        if not path.is_file() or path.stat().st_size <= 1000:
            return False
        with path.open("rb") as handle:
            return any(line.startswith(b"ATOM  ") for line in handle)
    except OSError:
        return False


def content_has_atom_record(content: bytes) -> bool:
    return any(line.startswith(b"ATOM  ") for line in content.splitlines())


def download_afdb(session: requests.Session, uid: str, save_path: Path) -> Tuple[bool, str]:
    if has_atom_record(save_path):
        return True, "cached"

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{AFDB_MODEL_VERSION}.pdb"
    temp_path: Optional[Path] = None
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        content = response.content
        if len(content) <= 1000 or not content_has_atom_record(content):
            return False, "downloaded content is not a valid PDB"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pdb", prefix=f"{uid}_", dir=save_path.parent, delete=False
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        os.replace(temp_path, save_path)
        return True, "downloaded"
    except requests.RequestException as exc:
        return False, f"request error: {exc}"
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def select_chain(model):
    chains = list(model.get_chains())
    if not chains:
        raise ValueError("structure contains no chains")
    for chain in chains:
        if chain.id == "A":
            return chain
    return chains[0]


def locate_existing_pdb(uid: str) -> Optional[Path]:
    candidate = AFDB_PDB_DIR / f"{uid}.pdb"
    if has_atom_record(candidate):
        return candidate
    return None


def build_residue_dict(chain) -> Dict[int, object]:
    residues: Dict[int, object] = {}
    for residue in chain.get_residues():
        hetflag, residue_number, insertion_code = residue.id
        if hetflag != " " or insertion_code.strip():
            continue
        if "CA" in residue:
            residues[int(residue_number)] = residue
    return residues


def parse_structure_and_dssp(pdb_path: Path, dssp_executable: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]
    chain = select_chain(model)
    residues = build_residue_dict(chain)
    if not residues:
        raise ValueError("selected chain contains no standard CA-bearing residues")

    try:
        # Use a temporary copy to avoid malformed DBREF records.
        with tempfile.TemporaryDirectory(prefix="ssg_dssp_") as directory:
            temporary_pdb = Path(directory) / pdb_path.name
            lines = pdb_path.read_bytes().splitlines(keepends=True)
            temporary_pdb.write_bytes(b"".join(
                line for line in lines
                if line[:6].strip() not in {b"DBREF", b"DBREF1", b"DBREF2"}
            ))
            dssp = DSSP(
                model, str(temporary_pdb),
                dssp=dssp_executable, file_type="PDB"
            )
        dssp_error = ""
    except Exception as exc:
        dssp = None
        dssp_error = f"DSSP failed: {exc}"
    return chain, residues, dssp, dssp_error


def extract_window(
    uid: str,
    target_position: int,
    expected_sequence: str,
    chain,
    residues: Dict[int, object],
    dssp,
) -> Tuple[Dict[str, np.ndarray], str, int, int]:
    coords = np.zeros((WINDOW_SIZE, 3), dtype=np.float32)
    aa_ids = np.full((WINDOW_SIZE,), AA_UNKNOWN_ID, dtype=np.int64)
    ss_ids = np.full((WINDOW_SIZE,), SS_UNKNOWN_ID, dtype=np.int64)
    rsa_vals = np.zeros((WINDOW_SIZE, 1), dtype=np.float32)
    valid_mask = np.zeros((WINDOW_SIZE,), dtype=np.float32)
    dssp_mask = np.zeros((WINDOW_SIZE,), dtype=np.float32)
    residue_ids = np.arange(
        target_position - CENTER_INDEX,
        target_position + (WINDOW_SIZE - CENTER_INDEX),
        dtype=np.int32,
    )

    if len(expected_sequence) != WINDOW_SIZE:
        return {
            "coords": coords,
            "aa_ids": aa_ids,
            "ss_ids": ss_ids,
            "rsa_vals": rsa_vals,
            "valid_mask": valid_mask,
            "dssp_mask": dssp_mask,
            "residue_ids": residue_ids,
        }, f"invalid Sequence length: {len(expected_sequence)}", 0, 0

    if expected_sequence[CENTER_INDEX] != "K":
        return {
            "coords": coords,
            "aa_ids": aa_ids,
            "ss_ids": ss_ids,
            "rsa_vals": rsa_vals,
            "valid_mask": valid_mask,
            "dssp_mask": dssp_mask,
            "residue_ids": residue_ids,
        }, "Sequence position 26 is not K", 0, 0

    target_residue = residues.get(target_position)
    if target_residue is None:
        return {
            "coords": coords,
            "aa_ids": aa_ids,
            "ss_ids": ss_ids,
            "rsa_vals": rsa_vals,
            "valid_mask": valid_mask,
            "dssp_mask": dssp_mask,
            "residue_ids": residue_ids,
        }, "target residue is absent from PDB", 0, 0
    if AA3_TO_1.get(target_residue.resname.strip().upper(), "X") != "K":
        return {
            "coords": coords,
            "aa_ids": aa_ids,
            "ss_ids": ss_ids,
            "rsa_vals": rsa_vals,
            "valid_mask": valid_mask,
            "dssp_mask": dssp_mask,
            "residue_ids": residue_ids,
        }, "PDB target residue is not LYS", 0, 0

    mismatch_count = 0
    dssp_valid_count = 0
    for local_index, residue_number in enumerate(residue_ids.tolist()):
        residue = residues.get(residue_number)
        if residue is None:
            continue

        aa = AA3_TO_1.get(residue.resname.strip().upper(), "X")
        expected_aa = expected_sequence[local_index]
        if expected_aa != "X" and aa != expected_aa:
            mismatch_count += 1
            continue

        coords[local_index] = np.asarray(residue["CA"].coord, dtype=np.float32)
        aa_ids[local_index] = AA_TO_ID.get(aa, AA_UNKNOWN_ID)
        valid_mask[local_index] = 1.0

        if dssp is None:
            continue
        dssp_key = (chain.id, residue.id)
        if dssp_key not in dssp:
            continue

        record = dssp[dssp_key]
        ss_char = str(record[2]).strip() or "-"
        relative_asa = record[3]
        try:
            relative_asa = float(relative_asa)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(relative_asa):
            continue

        ss_ids[local_index] = DSSP_TO_ID.get(ss_char, SS_UNKNOWN_ID)
        rsa_vals[local_index, 0] = np.float32(np.clip(relative_asa, 0.0, 1.0))
        dssp_mask[local_index] = 1.0
        dssp_valid_count += 1

    valid_count = int(valid_mask.sum())
    if mismatch_count:
        status = f"sequence mismatch at {mismatch_count} residue(s)"
    elif valid_count == WINDOW_SIZE and dssp_valid_count == WINDOW_SIZE:
        status = "ok"
    elif valid_count == WINDOW_SIZE:
        status = f"partial DSSP: {dssp_valid_count}/{WINDOW_SIZE}"
    else:
        status = f"partial coordinates: {valid_count}/{WINDOW_SIZE}; DSSP: {dssp_valid_count}/{WINDOW_SIZE}"

    return {
        "coords": coords,
        "aa_ids": aa_ids,
        "ss_ids": ss_ids,
        "rsa_vals": rsa_vals,
        "valid_mask": valid_mask,
        "dssp_mask": dssp_mask,
        "residue_ids": residue_ids,
    }, status, valid_count, dssp_valid_count


def validate_dataframe(df: pd.DataFrame) -> None:
    required = {"UniProt ID", "Position", "Sequence"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"DATA_PATH is missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("DATA_PATH contains no rows")
    if df["UniProt ID"].map(normalize_uid).eq("").any():
        raise ValueError("UniProt ID contains missing or empty values")
    positions = pd.to_numeric(df["Position"], errors="raise")
    valid_positions = (
        positions.notna()
        & (positions > 0)
        & (positions % 1 == 0)
        & (positions <= np.iinfo(np.int32).max - WINDOW_SIZE)
    )
    if not valid_positions.all():
        raise ValueError("Position must contain positive integers within the int32 range")


def configure_paths() -> None:
    names = ('DATA_PATH', 'AFDB_PDB_DIR', 'OUTPUT_NPZ', 'STATUS_CSV',)
    missing = [name for name in names if not str(globals()[name]).strip()]
    if missing:
        raise ValueError("Set the following paths before running: " + ", ".join(missing))
    for name in names:
        globals()[name] = Path(globals()[name]).expanduser()

def main() -> None:
    configure_paths()
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH does not exist: {DATA_PATH}")

    dssp_executable = require_mkdssp()
    AFDB_PDB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    STATUS_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(DATA_PATH)
    validate_dataframe(df)
    n_samples = len(df)
    uniprot_ids = df["UniProt ID"].map(normalize_uid).to_numpy(dtype=str)
    target_positions = pd.to_numeric(df["Position"], errors="raise").to_numpy(dtype=np.int32)

    coords_all = np.zeros((n_samples, WINDOW_SIZE, 3), dtype=np.float32)
    aa_ids_all = np.full((n_samples, WINDOW_SIZE), AA_UNKNOWN_ID, dtype=np.int64)
    ss_ids_all = np.full((n_samples, WINDOW_SIZE), SS_UNKNOWN_ID, dtype=np.int64)
    rsa_vals_all = np.zeros((n_samples, WINDOW_SIZE, 1), dtype=np.float32)
    valid_mask_all = np.zeros((n_samples, WINDOW_SIZE), dtype=np.float32)
    dssp_mask_all = np.zeros((n_samples, WINDOW_SIZE), dtype=np.float32)
    residue_ids_all = np.zeros((n_samples, WINDOW_SIZE), dtype=np.int32)
    sample_keys = np.empty((n_samples,), dtype=f"<U{max(32, int(df['UniProt ID'].astype(str).str.len().max()) + 24)}")

    status_rows = [None] * n_samples
    total_valid_nodes = 0
    total_dssp_nodes = 0

    working_df = df.copy()
    working_df["_uid"] = working_df["UniProt ID"].map(normalize_uid)

    session = requests.Session()
    session.headers.update({"User-Agent": "SSG-Kla-51mer-feature-builder/1.0"})

    grouped = working_df.groupby("_uid", sort=False, dropna=False)
    for uid, group in tqdm(grouped, total=working_df["_uid"].nunique(dropna=False), desc="Proteins"):
        uid = str(uid).strip()
        if not uid:
            for row_index in group.index:
                sample_keys[row_index] = "invalid|missing_uid"
                status_rows[row_index] = {
                    "row_index": int(row_index), "sample_key": "invalid|missing_uid", "status": "missing UniProt ID",
                    "valid_nodes": 0, "dssp_nodes": 0,
                }
            continue

        pdb_path = locate_existing_pdb(uid)
        if pdb_path is None:
            pdb_path = AFDB_PDB_DIR / f"{uid}.pdb"
            downloaded, download_status = download_afdb(session, uid, pdb_path)
        else:
            downloaded, download_status = True, f"cached: {pdb_path}"
        if not downloaded:
            for row_index in group.index:
                try:
                    position = int(working_df.at[row_index, "Position"])
                    key = sample_key(uid, position)
                except (TypeError, ValueError):
                    key = f"{uid}|invalid"
                sample_keys[row_index] = key
                status_rows[row_index] = {
                    "row_index": int(row_index), "sample_key": key,
                    "status": f"PDB unavailable: {download_status}", "valid_nodes": 0, "dssp_nodes": 0,
                }
            continue

        try:
            chain, residues, dssp, dssp_error = parse_structure_and_dssp(pdb_path, dssp_executable)
        except Exception as exc:
            for row_index in group.index:
                try:
                    position = int(working_df.at[row_index, "Position"])
                    key = sample_key(uid, position)
                except (TypeError, ValueError):
                    key = f"{uid}|invalid"
                sample_keys[row_index] = key
                status_rows[row_index] = {
                    "row_index": int(row_index), "sample_key": key,
                    "status": f"PDB parse failed: {exc}", "valid_nodes": 0, "dssp_nodes": 0,
                }
            continue

        for row_index in group.index:
            try:
                target_position = int(working_df.at[row_index, "Position"])
            except (TypeError, ValueError):
                status_rows[row_index] = {
                    "row_index": int(row_index), "sample_key": f"{uid}|invalid",
                    "status": "invalid Position", "valid_nodes": 0, "dssp_nodes": 0,
                }
                sample_keys[row_index] = f"{uid}|invalid"
                continue

            key = sample_key(uid, target_position)
            sample_keys[row_index] = key
            expected_sequence = normalize_sequence(working_df.at[row_index, "Sequence"])
            features, status, valid_count, dssp_count = extract_window(
                uid, target_position, expected_sequence, chain, residues, dssp
            )

            coords_all[row_index] = features["coords"]
            aa_ids_all[row_index] = features["aa_ids"]
            ss_ids_all[row_index] = features["ss_ids"]
            rsa_vals_all[row_index] = features["rsa_vals"]
            valid_mask_all[row_index] = features["valid_mask"]
            dssp_mask_all[row_index] = features["dssp_mask"]
            residue_ids_all[row_index] = features["residue_ids"]
            total_valid_nodes += valid_count
            total_dssp_nodes += dssp_count

            if dssp_error:
                status = f"{status}; {dssp_error}"
            status_rows[row_index] = {
                "row_index": int(row_index), "sample_key": key, "status": status,
                "valid_nodes": valid_count, "dssp_nodes": dssp_count,
            }

    if any(row is None for row in status_rows):
        missing_rows = [i for i, row in enumerate(status_rows) if row is None]
        raise RuntimeError(f"internal error: {len(missing_rows)} rows were not processed")

    metadata = np.array(
        [
            "window_size=51",
            "center_index_zero_based=25",
            "coordinate_atom=CA",
            "rsa_kind=DSSP_relative_solvent_accessibility",
            "ss_unknown_id=8",
            f"afdb_model_version={AFDB_MODEL_VERSION}",
        ],
        dtype="U256",
    )

    temporary_npz = OUTPUT_NPZ.with_name(OUTPUT_NPZ.name + ".tmp.npz")
    np.savez_compressed(
        temporary_npz,
        coords=coords_all,
        aa_ids=aa_ids_all,
        ss_ids=ss_ids_all,
        rsa_vals=rsa_vals_all,
        valid_mask=valid_mask_all,
        dssp_mask=dssp_mask_all,
        residue_ids=residue_ids_all,
        uniprot_ids=uniprot_ids,
        target_positions=target_positions,
        center_index=np.array(CENTER_INDEX, dtype=np.int32),
        window_size=np.array(WINDOW_SIZE, dtype=np.int32),
        sample_keys=sample_keys,
        metadata=metadata,
    )
    os.replace(temporary_npz, OUTPUT_NPZ)

    status_df = pd.DataFrame(status_rows).sort_values("row_index")
    status_df.to_csv(STATUS_CSV, index=False)

    expected_nodes = n_samples * WINDOW_SIZE
    complete_coord_samples = int((valid_mask_all.sum(axis=1) == WINDOW_SIZE).sum())
    complete_dssp_samples = int((dssp_mask_all.sum(axis=1) == WINDOW_SIZE).sum())
    print(f"Saved: {OUTPUT_NPZ}")
    print(f"Status report: {STATUS_CSV}")
    print(f"Samples: {n_samples}")
    print(f"Complete 51-coordinate samples: {complete_coord_samples}/{n_samples}")
    print(f"Complete 51-DSSP samples: {complete_dssp_samples}/{n_samples}")
    print(f"Valid coordinate nodes: {total_valid_nodes}/{expected_nodes}")
    print(f"Valid DSSP nodes: {total_dssp_nodes}/{expected_nodes}")


if __name__ == "__main__":
    main()
