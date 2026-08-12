# MaAI API Documentation

[GitHub Repository (MaAI-Kyoto/MaAI)](https://github.com/MaAI-Kyoto/MaAI)

Welcome to the official API documentation for MaAI (Real-time and Continuous Non-Linguistic Behavior Generation Software).

This documentation provides detailed specifications for each module, class, and function included in the MaAI source code.
You can browse the detailed API references for each module using the top navigation bar or the left menu.

## Core Modules

* **[`maai.encoder`](api/encoder.md)**: Modules related to encoding audio and text features.
* **[`maai.model`](api/model.md)**: Core modules for model building.
* **[`maai.input`](api/input.md)**: Modules for input data processing and management.
* **[`maai.output`](api/output.md)**: Modules for output data generation and management.
* **[`maai.objective`](api/objective.md)**: Modules defining objective functions, including loss functions for optimization.
* **[`maai.util`](api/util.md)**: A collection of utility functions.
* **`maai.models`**: Specific model architecture definitions corresponding to various tasks and conditions (see below).

## Models

Each model is selected through the `mode` argument of the `Maai` class.

* **[`maai.models.config`](api/models/config.md)**: `VapConfig`, the configuration object shared by all models.
* **[`maai.models.vap`](api/models/vap.md)** (`mode="vap"`, `"vap_mc"`): Voice Activity Projection, the turn-taking model that predicts who will be speaking in the near future. `"vap_mc"` is the noise-robust (multi-condition) variant.
* **[`maai.models.vap_mono`](api/models/vap_mono.md)** (`mode="vap_mono"`): Single-channel variant of VAP; the outputs are converted to single float values for the one input channel.
* **[`maai.models.vad`](api/models/vad.md)** (`mode="vad"`, `"vad_mono"`): Voice Activity Detection. Unlike the turn-taking models it does not predict the future, but detects whether each participant is speaking *right now*. Both channels are processed jointly with cross-channel attention, so the model can tell who is actually talking even when one speaker's voice leaks into the other's microphone (crosstalk). `"vad_mono"` is the single-channel variant.
* **[`maai.models.vap_bc`](api/models/vap_bc.md)** (`mode="bc"`): Backchannel prediction.
* **[`maai.models.vap_bc_2type`](api/models/vap_bc_2type.md)** (`mode="bc_2type"`): Backchannel prediction distinguishing two backchannel types.
* **[`maai.models.vap_nod`](api/models/vap_nod.md)** (`mode="nod"`): Head nod generation.
* **[`maai.models.vap_nod_para`](api/models/vap_nod_para.md)** (`mode="nod_para"`): Head nod generation with nod parameters.
* **[`maai.models.vap_prompt`](api/models/vap_prompt.md)** (`mode="vap_prompt"`): VAP conditioned on a text prompt.

Usage guides for each mode (supported languages, frame rates, and sample code) are available in the
[repository README](https://github.com/MaAI-Kyoto/MaAI/blob/main/README.md).

## Usage

Select the module you wish to learn more about from the navigation menu on the left.
Each page is automatically generated from the Docstrings within the source code.
