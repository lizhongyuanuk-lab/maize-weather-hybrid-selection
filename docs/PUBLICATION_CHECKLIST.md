# Publication boundary and release checklist

## Current state

This project is an independent Git repository, separate from its enclosing
workspace, which contains unrelated personal application material. Only this
project directory is the GitHub publication boundary; never push the enclosing
workspace. A local commit or configured remote does not itself establish that
the files have been uploaded: verify the remote repository after pushing.

The ignore file is a safeguard, not upload permission or history removal. It
does not protect already tracked or force-added files. The checker includes
tracked files as well as non-ignored untracked candidates.

## Lightweight repository contents

- Research code, synthetic tests, protocols and dependency manifests.
- README, data/method/reproduction notes.
- Small aggregate analysis CSVs, including negative and mixed results.
- The current manuscript source, figures and PDF. The separate website source
  and its third-party runtime are not part of this code release.

Keep raw observations, prepared labels, feature matrices, serialized models,
row-level prediction archives, runtimes and generated logs outside the code
repository. No Git LFS configuration was introduced; distributing large
artifacts is a separate decision.

## Preserve provenance without local identifiers

Immutable local manifests, unit records and audit receipts contain absolute
paths and machine-specific provenance. They are excluded from publication
candidates. **Do not redact them in place:** that breaks their hashes and the
completed run's identity.

For distribution, create a separate reviewed, path-redacted release copy with
its own checksum and a documented mapping to the original manifest hash. Never
claim those redacted bytes are the original artifact. This task does not create
that archive or certify aggregate CSVs alone as sufficient for full replay.

Review notebooks, HTML/PDF, frontend snapshots, runtime metadata, browser/thread
links, screenshots and downloads as well as source text. Prefer relative source
paths to author-machine paths.

## Rights and attribution decisions

- The official G2F catalog lists PDDL for its dataset; retain its citation and
  review resource notes before redistribution.
- The owner chose a public repository without a code license for now. See
  [CODE_RIGHTS.md](CODE_RIGHTS.md). Public visibility is not an open-source
  license or permission to reuse the original code.
- Third-party frontend/runtime code, fonts, icons, images and dependencies keep
  their own terms. A root license cannot relicense them. Check origin/notices.
- Do not invent authors, affiliations, DOI, publication acceptance, institutional
  endorsement or approval. This is an independent retrospective project, not
  a published method or operational agricultural product.

## Preflight

Run from the project directory:

    python scripts/check_publication.py --self-test
    python scripts/check_publication.py

The checker reads Git candidates and checks boundary, large files, symlinks,
machine paths, high-signal secret patterns and private artifact types. It prints
relative file/line locations, not matching secret values. It checks local
README/document links where possible. It does not scan Git history, prove every
secret absent, determine third-party rights, or publish anything.
A nonzero exit is a **review stop**, not a reason to force-add files.

Before each public push:

1. Confirm the repository root is exactly this project, not its parent.
2. Confirm the destination account/repository and its public visibility;
   visibility does not justify including unrelated personal material.
3. Keep the no-license decision explicit and review third-party
   redistribution rights for any newly added assets.
4. Resolve preflight findings; inspect the exact staged diff and presentation
   snapshots. Do not upload the entire parent workspace.
5. Confirm README numbers match aggregate CSVs. Retain the unsuccessful
   segmented-weather result and retrospective caveats.
6. Decide separately whether to distribute an audit/model archive, with data
   rights, redaction and reproducibility limits stated.
7. Verify the resulting GitHub file list and README after publishing.

## Remaining portability boundary

The guardrails do not convert historical absolute-path manifests into portable
ones. Full new-machine rolling/segmented reproduction still needs deliberate
versioned provenance migration. Original 2024-only notebook/PDF presentations
remain historical unless explicitly updated; their narrower scope is not the
latest five-year study.
