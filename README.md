# Project Kahn: Strategic Interaction Between Frontier AI Models

This repository contains the simulation code and tournament data for "Strategic Interaction Between Frontier AI Models: A Computational Approach to Crisis Escalation".

## Paper

**AI Arms and Influence: Frontier Models Exhibit Sophisticated Reasoning in Simulated Nuclear Crises**

The paper is available on arXiv: [link to be added]

## Repository Contents

### Game Code
- `Kahn_game_v11.py` - Open-ended scenario variant (no explicit deadline)
- `Kahn_game_v12.py` - Deadline scenario variant (explicit time pressure)
- `scenarios.py` - Scenario definitions and prompt generation
- `llm_providers.py` - Unified LLM provider layer (supports original + Chinese models)
- `test_providers.py` - API connectivity validation script
- `.env.example` - API key configuration template

### Configuration Files
- `state_a_leader_kahn.json` / `state_b_leader_kahn.json` - Leader personality profiles
- `state_a_military_kahn.json` / `state_b_military_kahn.json` - Military capabilities
- `state_a_assessment_kahn.json` / `state_b_assessment_kahn.json` - Intelligence assessments

### Tournament Data
The `tournament_results/` directory contains CSV logs from all 21 games in the tournament, featuring:
- **Claude Sonnet 4** (claude-sonnet-4-20250514)
- **GPT-5.2** (gpt-5.2)
- **Gemini 3 Flash** (gemini-3-flash-preview)

Each CSV contains turn-by-turn data including:
- Model actions and signals
- Territory changes
- Military attrition
- Full model reasoning/rationales

## Requirements

- Python 3.10+
- `pip install -r requirements.txt`
- API keys for at least 2 providers (see below)

## Supported Models

### Original Models
- **Claude Sonnet 4** (claude-sonnet-4-20250514) - Anthropic
- **GPT-5.2** (gpt-5.2) - OpenAI
- **Gemini 3 Flash** (gemini-3-flash-preview) - Google

### Chinese Domestic Models (New)
- **DeepSeek V3** (deepseek-chat) / **R1** (deepseek-reasoner) - [platform.deepseek.com](https://platform.deepseek.com)
- **Qwen** (qwen-max, qwen-plus) - [dashscope.console.aliyun.com](https://dashscope.console.aliyun.com)
- **GLM-4** (glm-4-plus, glm-4-flash) - [open.bigmodel.cn](https://open.bigmodel.cn)
- **Moonshot Kimi** (moonshot-v1-128k) - [platform.moonshot.cn](https://platform.moonshot.cn)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your API keys (need at least 2 providers)

# 3. Validate connectivity
python test_providers.py

# 4. Run a game
python Kahn_game_v11.py --model_a deepseek-chat --model_b qwen-max --turns 20 --scenario v7_alliance
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{payne2026arms,
  title={AI Arms and Influence: Frontier Models Exhibit Sophisticated Reasoning in Simulated Nuclear Crises},
  author={Payne, Kenneth},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Creative Commons Attribution-NonCommercial 4.0 International).

You are free to share and adapt this material for non-commercial purposes, provided you give appropriate credit. For commercial licensing inquiries, contact Kenneth Payne.

See [LICENSE](LICENSE) for details.

## Contact

Kenneth Payne - [institution/email to be added]
