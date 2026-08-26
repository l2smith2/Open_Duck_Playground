# Kaggle API batch runs

This repo supports two Kaggle paths:

- interactive notebook runs, which are best for persistence, manual review, and stopping at checkpoints;
- API batch runs, which are best for repeatable smoke and benchmark jobs without browser copy/paste.

The API path still uses Kaggle quota. It cannot stop an already-open interactive notebook session. Use Kaggle's Run > Stop session or Active Events for that.

## One-time local setup

Create a Kaggle API token from Kaggle account settings. Store it outside this repository.

Newer Kaggle accounts may provide a single access token. On Windows, save only the token text here:

    C:\Users\YOUR_WINDOWS_USER\.kaggle\access_token

Install and test the CLI:

    py -m pip install --user kaggle
    kaggle --version
    kaggle kernels list -m --page-size 5

If the kaggle command is not found, use:

    py -m kaggle kernels list -m --page-size 5

or add the Python user Scripts directory to PATH.

Never commit access_token, kaggle.json, environment-variable exports, or copied token text.

## Prepare only

To create the ignored Kaggle batch folder without spending GPU quota:

    START_KAGGLE_BATCH.cmd -NoPush

This writes user-specific files under .kaggle-api/, which is ignored by Git.

## Start smoke plus benchmark

To upload and start a private Kaggle script run:

    START_KAGGLE_BATCH.cmd

The default accelerator is NvidiaTeslaT4. To try P100 instead:

    START_KAGGLE_BATCH.cmd -Accelerator NvidiaTeslaP100

The batch script runs:

- setup with uv sync --locked;
- 1M smoke training;
- 20M nominal benchmark;
- compute projection;
- after_benchmark.zip output packaging.

Batch runs start from a clean Kaggle worker and do not use the interactive notebook's file persistence. A completed smoke test from an interactive session may still be useful evidence, but an API batch run normally repeats the smoke test before benchmarking.

## Check status

Use either the launcher option:

    START_KAGGLE_BATCH.cmd -NoPush -Poll

or the raw CLI:

    kaggle kernels status YOUR_KAGGLE_USER/open-duck-free-first-bdx-benchmark

## Download outputs

When the run completes:

    START_KAGGLE_BATCH.cmd -NoPush -Download

or:

    kaggle kernels output YOUR_KAGGLE_USER/open-duck-free-first-bdx-benchmark -p kaggle-outputs/open-duck-free-first-bdx-benchmark -o

Downloaded files stay under kaggle-outputs/, which is ignored by Git.

## Safety notes

- The public script template stays generic.
- Generated .kaggle-api files may contain your fork URL and Kaggle username, but not your token.
- Outputs, checkpoints, ONNX files, and ZIP bundles stay outside Git.
- Use interactive Kaggle when you need human reference-motion review or file-persistence resume behavior.
