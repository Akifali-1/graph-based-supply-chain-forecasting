# Security and artifact handling

## Never commit

- Google Drive OAuth token files, OAuth client JSON, cookies, or refresh tokens.
- Cloud credentials, API keys, passwords, private keys, or SSH keys.
- Model checkpoints, raw datasets, private logs, or downloaded Drive folders.

The reproducibility scripts accept private paths through command-line arguments;
they do not contain a default token path. The repository ignores common token,
credential, key, checkpoint, raw-data, and private Drive-download locations.

## Reproducible access

Keep the OAuth token outside the repository and pass it to
`download_drive_artifacts.py` with `--token-file`. Do not paste the token into
shell history, notebooks, source files, or issue reports. If a credential is
ever committed, revoke it immediately, remove it from all reachable history,
and replace it through the provider's credential-management interface.

## Pre-publication check

Before publishing, inspect the exact staged set:

```bash
git status --short
git diff --cached --check
git diff --cached --name-only
```

Confirm that only source, documentation, tests, and reviewed aggregate
`paper_outputs/` artifacts are staged. Checkpoint and Drive-download paths must
remain absent.
