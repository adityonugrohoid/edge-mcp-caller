# Real-World Use Cases for Edge Specialist Models

## The Pattern

The v0.1 filesystem tools (list_directory, read_file, search_files) are the proof of concept. The real finding is: **at 270M scale, specialization is the only approach that works — and for fixed-tool edge deployments, it's not just viable, it's superior to models 440x larger.**

Any scenario where:
- The tools are **stable** (don't change frequently)
- The device is **constrained** (no GPU, limited memory, no internet)
- **Privacy/latency/cost** matters

That's the specialist's territory.

## Use Case 1: Smart Home Hub (strongest)

**Tools**: `turn_light(room)`, `set_thermostat(temp)`, `check_lock(door)`

- Called 50+ times/day by every family member
- Runs on a $30 hub — no GPU, no cloud
- **Works when internet is down** — the #1 complaint about Alexa/Google Home
- Voice commands never leave the house — full privacy
- These 3 tools haven't changed in 20 years of home automation protocols
- 263ms response time vs 800ms+ cloud round-trip — the difference between "natural" and "laggy"

## Use Case 2: Factory Floor Sensor Monitor

**Tools**: `read_sensor(id)`, `list_sensors(zone)`, `search_alerts(pattern)`

- Ruggedized tablet on a factory floor — no WiFi allowed near heavy machinery
- Technicians ask natural language questions about equipment
- Same 3 tools for decades — industrial protocols don't change
- Zero API cost at scale — factories run 24/7, thousands of queries/day
- 291MB fits on any industrial tablet

## Use Case 3: Medical Bedside Monitor

**Tools**: `read_vitals(patient)`, `list_alarms(ward)`, `search_history(patient, metric)`

- **HIPAA requires on-device processing** — patient data can't go to cloud
- Nurses ask questions in natural language during rounds
- Called every few seconds per patient
- Latency matters — 263ms vs waiting for cloud during an emergency

## Why Specialist Wins Here

| Factor | Specialist (ours) | Generalist (cloud) |
|--------|-------------------|---------------------|
| Prompt overhead | 20 tokens | 264+ tokens |
| Latency | 263ms (local) | 800ms+ (network round-trip) |
| Internet required | No | Yes |
| Privacy | Data stays on device | Data sent to cloud |
| API cost | $0 | Per-request billing |
| Model size | 291MB | N/A (cloud-hosted) |
| Offline capable | Yes | No |
| Tool changes | Retrain (~1 hour) | Update prompt (instant) |

The trade-off is explicit: the specialist trades **flexibility** (can't add tools without retraining) for **efficiency** (13x fewer tokens, local inference, zero cost). For fixed-tool edge deployments, that trade-off is clearly worth it.

## The v0.1 Proof

- 90.8% combined accuracy at 270M scale
- Beats GPT-OSS 120B (23.3%) — a model 440x larger
- 100% MCP execution success across 360 queries
- 20 tokens/request vs 264 tokens/request (13x savings)
- 263ms end-to-end latency on a laptop

The filesystem tools prove the approach. The next step is applying it to one of these domains.
