<div align="center">

# 🌙 DreamEngine

**Generative Simulation and Memory Consolidation for Autonomous AI Agents**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Ray](https://img.shields.io/badge/orchestration-Ray-orange)](https://ray.io)
[![Framework](https://img.shields.io/badge/RL-DreamerV3%20%2B%20RLlib-green)](https://github.com/google-deepmind/dreamerv3)
[![Status](https://img.shields.io/badge/status-active%20development-yellow)]()

*DreamEngine gives autonomous AI agents the ability to dream — consolidating experience, simulating counterfactual futures, and improving policy offline without touching the real world.*

</div>

---

## Overview

DreamEngine is a production-grade framework for **generative simulation and memory consolidation** in autonomous AI agents. Inspired by biological sleep-wake cycles, it enables agents to learn from both real and imagined experience by decoupling online interaction from offline policy optimization.

Rather than learning only from costly real-world steps, DreamEngine agents:

- **Consolidate** episodic experience into tiered memory (hot → warm → cold)
- **Dream** — generate imagined trajectories in learned latent space via a world model
- **Evaluate** dream quality with fidelity scoring and constraint validation
- **Integrate** validated policy updates through a guarded, staged deployment gate

The canonical training ratio is **1024 dream steps per real environment step**, giving DreamEngine agents orders-of-magnitude more policy optimization from the same real-world interaction budget.

---

## Architecture

DreamEngine is structured as five functional layers, each with distinct latency, resource, and failure profiles:

```
┌──────────────────────────────────────────────────────────────────────┐
│                         DreamEngine Stack                            │
├──────────────┬───────────────────────────────────────────────────────┤
│ LAYER        │ COMPONENTS                                            │
├──────────────┼───────────────────────────────────────────────────────┤
│ Perception   │ EnvRunner actors · Feature extractors · Redis streams │
│ Memory       │ Redis (hot) · TimescaleDB (warm) · Qdrant (vector)    │
│ Generation   │ DreamerV3 RSSM · LiteLLM routing · Diffusion models  │
│ Simulation   │ Latent sandbox · Digital twin · Symbolic validators   │
│ Learning     │ LearnerGroup · EWC · KL constraints · Staged deploy   │
└──────────────┴───────────────────────────────────────────────────────┘
```

### The Dreaming Loop

```
Perceive → Consolidate → Imagine → Evaluate → Integrate
    ↑                                              │
    └──────────── policy feedback ─────────────────┘
```

| Stage | Biological Analog | Key Technology |
|---|---|---|
| **Perceive** | Wakeful experience encoding | EnvRunner, Redis Streams |
| **Consolidate** | Hippocampal-neocortical transfer | EpisodeReplayBuffer, TimescaleDB, Qdrant |
| **Imagine** | REM imagination / counterfactual simulation | DreamerV3 RSSM, LiteLLM, Diffusion models |
| **Evaluate** | Dream salience assessment, nightmare detection | Fidelity classifiers, uncertainty quantification, symbolic validators |
| **Integrate** | Memory reconsolidation | LearnerGroup, EWC, KL constraints, staged deployment |

---

## Core Components

### Memory Substrate — Tiered Storage

| Tier | Technology | Retention | Latency | Use |
|---|---|---|---|---|
| **Hot** | Redis + NVMe SSD | 0–7 days | < 1ms | Active ingest, real-time access |
| **Warm** | TimescaleDB + SSD | 7–90 days | 10–100ms | Episode retrieval, trend queries |
| **Cold** | Object Storage (S3) | 90 days–1 yr | 100–1000ms | Long-term archive |
| **Archive** | Glacier / Deep Archive | 1+ years | Minutes–hours | Compliance, replay |

Four memory modalities are maintained in parallel:
- **Episodic** — complete trajectories with temporal ordering (`EpisodeReplayBuffer`)
- **Semantic** — compressed knowledge graphs (Neo4j / Amazon Neptune)
- **Vector** — HNSW similarity search (Qdrant) for RAG-style dream seeding
- **Temporal** — causal chain indexing (TimescaleDB hypertables)

### World Model — DreamerV3 RSSM

The core generative backbone is a **Recurrent State-Space Model (RSSM)** with dual paths:

```
h_t = f_GRU(h_{t-1}, z_{t-1}, a_{t-1})   ← deterministic: predictable dynamics
z_t ~ p(z_t | h_t)                         ← stochastic: irreducible uncertainty
```

The world model supports four operations:

```python
encode(obs)                              → latent
step(latent, action)                     → next_latent, reward, done, info
decode(latent)                           → obs
imagine(initial_latent, policy, horizon) → trajectory
```

Model sizes range from XS (1M params) to XL (1B+), with distributed training via Ray `LearnerGroup`.

### Dream Scheduler — Intelligent Triggering

Dreams are triggered by four mechanisms:

| Trigger | Condition | Dream Type |
|---|---|---|
| **Idle** | GPU utilization < 15–20% | Micro-dream (H=5–15 steps) |
| **Entropy** | Policy uncertainty spike | Targeted exploration dream |
| **Surprise** | Reward prediction error > threshold | Generalization / adversarial dream |
| **Scheduled** | Kubernetes CronJob (e.g., nightly) | Deep consolidation dream |

Priority queue with preemption: P0 (critical / safety) → P1 (entropy/surprise) → P2 (idle) → P3 (scheduled background).

### Generative Engine — Pluggable Backends

| Backend | Domain | When to use |
|---|---|---|
| **RSSM / World Model** | Continuous state spaces | Default; fastest (10 k+ steps/sec on A100) |
| **LiteLLM-routed LLM** | Symbolic / language-mediated | Counterfactual reasoning, legal, text-based agents |
| **Latent Diffusion Models** | High-dimensional observation spaces | Robotics vision, audio, safety-critical visual validation |
| **Hybrid (LangGraph)** | Complex, multi-modal domains | Market stress tests, multi-step scenario composition |

### Policy Integration Gate — Staged Rollout

No dreamed policy update deploys automatically. All updates pass through a validation pipeline:

```
Shadow (0%, 1–24h) → Canary (1–10%, 1–7d) → Gradual (10–50%, 3–14d) → Full (100%)
```

Rollback triggers: performance degradation > 5%, safety constraint violation, anomaly alert, or human escalation. Elastic Weight Consolidation (EWC) protects critical parameters; KL-divergence constraints limit deviation from the wake-phase policy baseline.

---

## Multi-Agent Support

DreamEngine supports four dream-sharing models:

| Model | Privacy | Collective Learning |
|---|---|---|
| **Private** | Complete | None — agent-local only |
| **Shared Pool** | Reputation-weighted retrieval | High — cultural transmission via Qdrant embeddings |
| **Hierarchical** | Tactical private / strategic shared | Multi-level (individual → team → organization) |
| **Federated** | (ε, δ)-DP via gradient clipping + noise | Privacy-preserving shared world model updates |

Cross-framework dream exchange uses a standardized **MCP schema** (`dream://agent_id/dream_id` URIs), exposing generation and validation as callable MCP tools and enabling interoperability across heterogeneous agent stacks (LangChain, ElizaOS, CrewAI, custom).

---

## Hermes–OpenClaw Integration

DreamEngine is designed for the Hermes–OpenClaw agent architecture:

| Role | Agent | DreamEngine Responsibilities |
|---|---|---|
| **Supervisor** | Hermes | Scheduler policy · Dream admission control · Validation decisions · Rollout gate |
| **Worker** | OpenClaw / NemoClaw | Memory jobs · Latent rollout generation · Validator runs · Deploy tasks |

Workers are stateless; all dream state and policy provenance is managed by Hermes through structured protocol messages.

---

## Infrastructure

DreamEngine runs on Kubernetes with Ray for distributed execution.

### Namespace Isolation

| Namespace | Purpose | Network Policy |
|---|---|---|
| `dreamengine-wake` | Real-time inference, environment interaction | Egress to production services only |
| `dreamengine-dream` | Offline simulation, learning | **No egress to production — isolated** |
| `dreamengine-validate` | Staged policy validation | Egress to staging only |
| `dreamengine-shared` | Monitoring, config | Limited egress |

### Key Infrastructure Components

- **Ray** — `DreamWorker` actors, `EnvRunner` actors, `LearnerGroup`, distributed object store (zero-copy tensor sharing)
- **Redis** — Hot memory, dream event streams, scheduler heartbeats, pub/sub triggers
- **TimescaleDB** — Hypertable episode storage, continuous aggregation, causal chain indexing
- **Qdrant** — HNSW vector similarity search, hybrid dense+sparse retrieval, shared dream pool
- **LiteLLM** — Unified multi-provider LLM routing with fallback chains and budget tracking
- **Prefect / Dagster** — Async dream pipeline orchestration with retry and caching
- **Prometheus + Grafana** — Dream quality metrics, GPU utilization, scheduler telemetry

### GPU Allocation

| Worker Class | GPUs | Use Case |
|---|---|---|
| `LightweightDreamWorker` | 0.25 (fractional) | Latent imagination, small world models |
| `StandardDreamWorker` | 1.0 | Standard RSSM inference |
| `HeavyDreamWorker` | 4.0 | Large world models, diffusion generation |

---

## Security and Alignment

### Core Safety Guarantees

1. **Causal isolation** — dream pods have no network egress to production; actions taken in dreams cannot affect real systems
2. **Provenance tagging** — every memory, trajectory, and policy update carries `dream_id`, `agent_id`, `world_model_version`, and `fidelity_level`
3. **Validation gates** — no dreamed policy auto-deploys; shadow/canary evaluation is mandatory
4. **Human oversight hooks** — approval workflows for novel dream types, large policy changes, and anomaly escalation

### Known Risk Classes and Mitigations

| Risk | Mechanism | Mitigation |
|---|---|---|
| Confirmation bias loops | World model generates self-confirming scenarios | Adversarial dream generation, explicit failure replay, diversity objectives |
| Reality drift | Dream-policy cycles compound errors | MMD monitoring, held-out real evaluation, world model retraining triggers |
| Epistemic isolation | Agent confuses dreamed vs. real memory | Explicit provenance tags, importance-weighted discount for dream data |
| Dream pool poisoning | Byzantine agents inject harmful dreams | Byzantine-resilient aggregation (trimmed mean), outlier detection, reputation scoring |
| Reward hacking | Policy exploits simulation loopholes | Transfer validation, real-environment A/B testing, conservative Q-learning (CQL) |
| Alignment drift | Dreamed scenarios violate specified values | Value-embedding constraints, multi-objective reward shaping, symbolic validators |

---

## Use Cases

DreamEngine is domain-agnostic. Concrete application patterns from the architecture spec:

- **Autonomous Trading** — Overnight market stress-test dreams; counterfactual trade analysis; multi-desk shared world models with nightly synchronization
- **Robotics** — Sim-to-real transfer via residual dynamics dreaming; adversarial failure-mode exploration; MPC with latent trajectory optimization
- **Legal Reasoning Agents** — Hypothetical case outcome synthesis; adversarial courtroom simulation; regulatory impact forecasting
- **Infrastructure / Mining Optimization** — Energy-thermal co-simulation; hardware degradation forecasting; market-operational joint dreaming

---

## Development Roadmap

| Phase | Focus | Status |
|---|---|---|
| **Phase 1** | Single-agent MVP: episode ingest, Redis/TSDB/Qdrant, RSSM, latent rollouts, offline evaluation | 🔨 In development |
| **Phase 2** | Dream Scheduler, uncertainty scoring, fidelity validation, rollout gate, rollback | 🗂 Planned |
| **Phase 3** | Digital twin integration, symbolic validators, staged deployment pipeline | 🗂 Planned |
| **Phase 4** | Multi-agent sharing, MCP schema, federated privacy controls, cross-agent orchestration | 🗂 Planned |

---

## Repository Structure

```
DreamEngine/
├── dreamengine/
│   ├── memory/               # Redis, TimescaleDB, Qdrant substrate
│   ├── scheduler/            # Dream trigger logic and resource arbitration
│   ├── generation/           # World model (RSSM), LLM routing, diffusion backends
│   ├── simulation/           # Latent sandbox and digital twin environments
│   ├── learning/             # LearnerGroup, EWC, policy update pipeline
│   ├── orchestration/        # Ray actor definitions, DreamWorker, EnvRunner
│   ├── validation/           # Fidelity classifiers, symbolic validators, gates
│   └── schemas/              # Pydantic data contracts, MCP dream schema
├── k8s/
│   ├── namespaces/           # Wake / dream / validate / shared isolation
│   ├── cronjobs/             # Scheduled deep-dream consolidation jobs
│   └── helm/                 # DreamEngine Helm chart
├── configs/
│   ├── values-production.yaml
│   ├── values-staging.yaml
│   └── values-edge.yaml
├── tests/
├── docs/
│   └── architecture.md       # Full technical specification
└── README.md
```

---

## Quick Start

> **Note:** Phase 1 implementation is actively in development. The following reflects the intended interface.

### Prerequisites

- Python 3.11+
- Ray 2.x
- Redis 7+
- TimescaleDB (PostgreSQL 15+)
- Qdrant
- CUDA-capable GPU (recommended for world model training)

### Installation

```bash
git clone https://github.com/enuno/DreamEngine.git
cd DreamEngine
pip install -e ".[dev]"
```

### Start the memory substrate

```bash
# Redis
docker run -d -p 6379:6379 redis:7-alpine

# TimescaleDB
docker run -d -p 5432:5432 \
  -e POSTGRES_PASSWORD=dream \
  timescale/timescaledb:latest-pg15

# Qdrant
docker run -d -p 6333:6333 qdrant/qdrant
```

### Launch a DreamWorker pool

```python
import ray
from dreamengine.orchestration import create_dream_worker_pool
from dreamengine.schemas import DreamConfig

ray.init()

config = DreamConfig(
    imagination_horizon=15,
    num_dream_sequences=16,
    fidelity_level=1,  # 0=real, 1=latent, 2=decoded, 3=twin
    device="cuda"
)

workers = create_dream_worker_pool(
    num_workers=4,
    world_model_config={...},
    actor_checkpoint="path/to/actor.pt",
    critic_checkpoint="path/to/critic.pt",
    config=config
)
```

### Deploy via Helm

```bash
helm install dreamengine ./k8s/helm \
  -f k8s/helm/values-production.yaml \
  --namespace dreamengine
```

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Licensed under the [Apache License 2.0](LICENSE).

---

## References

- [DreamEngine Technical Architecture Specification](docs/architecture.md)
- [DreamerV3 — Mastering Diverse Domains World Models without Curriculum](https://arxiv.org/abs/2301.04104) — Hafner et al., 2023
- [Ray RLlib](https://docs.ray.io/en/latest/rllib/) — Distributed Reinforcement Learning
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io) — Cross-agent tool interoperability
- [Qdrant](https://qdrant.tech) — Vector similarity search
- [TimescaleDB](https://www.timescale.com) — Time-series PostgreSQL
- [LiteLLM](https://litellm.ai) — Unified LLM provider routing

---

<div align="center">
<sub>Built for the Hermes–OpenClaw multi-agent architecture · Designed for production · Aligned by construction</sub>
</div>
