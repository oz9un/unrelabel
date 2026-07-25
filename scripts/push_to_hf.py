#!/usr/bin/env python3
"""Publish the built unrelabel datasets to the Hugging Face Hub.

Auth is *yours*: run ``hf auth login`` first so your token lives in your own
environment. This script never takes a token as an argument and never prints one.

Usage:
    hf auth login                 # you enter your token here, not the script
    python scripts/build_hf_datasets.py   # build hf_export/ (offline)
    python scripts/push_to_hf.py --user <your-hf-username>            # dry run (prints plan)
    python scripts/push_to_hf.py --user <your-hf-username> --push     # actually upload

Defaults to PRIVATE repos (poisoned data): review on the Hub, then flip to public in
the repo settings, or pass --public to create them public from the start.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "hf_export"

# Local folder -> repo name suffix. The final repo id is "<user>/<repo>".
REPOS = {
    "unrelabel-demos": "unrelabel-demos",
    "unrelabel-poison-benchmark": "unrelabel-poison-benchmark",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default=None,
                    help="Repo owner. Defaults to your authenticated HF username; pass an org name to publish under an org.")
    ap.add_argument("--which", choices=["demos", "benchmark", "all"], default="all")
    ap.add_argument("--public", action="store_true", help="Create public repos (default: private)")
    ap.add_argument("--push", action="store_true", help="Actually upload (omit for a dry run)")
    args = ap.parse_args()

    if not EXPORT.is_dir():
        print(f"error: {EXPORT} not found. Run scripts/build_hf_datasets.py first.", file=sys.stderr)
        return 2

    wanted = {
        "demos": ["unrelabel-demos"],
        "benchmark": ["unrelabel-poison-benchmark"],
        "all": list(REPOS),
    }[args.which]

    private = not args.public
    owner = args.user or "<your-namespace>"
    print(f"Plan: publish {len(wanted)} dataset repo(s) to '{owner}' "
          f"({'PRIVATE' if private else 'PUBLIC'}){' [DRY RUN]' if not args.push else ''}\n")
    for folder in wanted:
        local = EXPORT / folder
        n = sum(1 for _ in local.rglob('*') if _.is_file())
        print(f"  {folder}  ->  {owner}/{REPOS[folder]}   ({n} files)")

    if not args.push:
        print("\nDry run. Re-run with --push to upload. "
              "Make sure you ran `hf auth login` first (with a WRITE token).")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("error: huggingface_hub not installed. `pip install -U huggingface-hub`.", file=sys.stderr)
        return 2

    api = HfApi()
    try:
        who = api.whoami()  # verifies you are logged in; uses your stored token
    except Exception:
        print("error: not logged in. Run `hf auth login` first (needs a WRITE token).", file=sys.stderr)
        return 2

    me = who.get("name", "?")
    role = (who.get("auth", {}) or {}).get("accessToken", {}).get("role")
    print(f"\nAuthenticated as: {me}" + (f" (token role: {role})" if role else "") + "\n")
    if role == "read":
        print("error: your token is READ-only, so it cannot create repos. Make a WRITE token at\n"
              "  https://huggingface.co/settings/tokens  then `hf auth logout && hf auth login`.",
              file=sys.stderr)
        return 2

    owner = args.user or me
    if args.user and args.user != me:
        print(f"note: publishing under '{args.user}', but you are '{me}'. That only works if\n"
              f"      '{args.user}' is an organization you have write access to.\n")

    for folder in wanted:
        local = EXPORT / folder
        repo_id = f"{owner}/{REPOS[folder]}"
        print(f"Uploading {folder} -> {repo_id} ...")
        try:
            api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        except Exception as e:
            print(f"error: could not create '{repo_id}': {e}\n"
                  f"  - is '{owner}' your username (see 'Authenticated as' above) or an org you can write to?\n"
                  f"  - does your token have WRITE / create-repo permission?",
                  file=sys.stderr)
            return 2
        api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(local),
                          commit_message="Publish unrelabel dataset")
        print(f"  done: https://huggingface.co/datasets/{repo_id}")
    print("\nPublished. If private, flip to public in each repo's Settings when ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
