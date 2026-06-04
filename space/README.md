---
title: Gemma4Tunnel Demo
emoji: 🚇
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: other
---

# Gemma4Tunnel Demo

A Gradio demo Space for the `TUSTResearcher/gemma4tunnel` LoRA adapter.

The Space loads:

```python
BASE_MODEL = "google/gemma-4-12B-it"
ADAPTER_MODEL = "TUSTResearcher/gemma4tunnel"
```

For a 12B model, choose GPU hardware or change the UI to call a Modal inference endpoint.
