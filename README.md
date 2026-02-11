# Project Kahn: Strategic Interaction Between Frontier AI Models

This repository contains the simulation code and tournament data for "Strategic Interaction Between Frontier AI Models: A Computational Approach to Crisis Escalation".

## Paper

The paper is available on arXiv: [link to be added]

## Repository Contents

### Game Code
- `Kahn_game_v11.py` - Open-ended scenario variant (no explicit deadline)
- `Kahn_game_v12.py` - Deadline scenario variant (explicit time pressure)
- `scenarios.py` - Scenario definitions and prompt generation

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
- API keys for Anthropic, OpenAI, and Google (Gemini)

## Citation

If you use this code or data, please cite:

```bibtex
@article{payne2026strategic,
  title={Strategic Interaction Between Frontier AI Models: A Computational Approach to Crisis Escalation},
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
