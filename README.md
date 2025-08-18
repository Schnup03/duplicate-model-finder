# duplicate-model-finder

Extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that scans model directories for duplicate files and allows deleting them directly from the UI.

## Features

- Scans common model directories (`models/Stable-diffusion`, `models/Lora`, `models/VAE`)
- Computes SHA256 hashes to identify identical files
- Presents duplicates in a dedicated tab with labeled buttons to scan and remove files

## Installation

Clone this repository into the `extensions` folder of your WebUI installation:

```bash
git clone <repo_url> extensions/duplicate-model-finder
```

## Usage

1. Launch the WebUI.
2. Open the **Duplicate Models** tab.
3. Click **Scan for duplicates**.
4. Select redundant files and press **Delete selected**.
