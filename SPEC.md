# DreamEngine — Technical Specification

**Version:** 0.1.0-draft  
**Status:** Active Development  
**Repository:** [github.com/enuno/DreamEngine](https://github.com/enuno/DreamEngine)  
**License:** Apache 2.0

---

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [System Overview](#2-system-overview)
3. [Formal Definitions](#3-formal-definitions)
4. [Data Contracts](#4-data-contracts)
5. [Component Specifications](#5-component-specifications)
   - 5.1 [Memory Substrate](#51-memory-substrate)
   - 5.2 [Dream Scheduler](#52-dream-scheduler)
   - 5.3 [Generative Engine](#53-generative-engine)
   - 5.4 [Simulation Environment](#54-simulation-environment)
   - 5.5 [Policy Update Loop](#55-policy-update-loop)
   - 5.6 [Orchestration Layer](#56-orchestration-layer)
6. [API Reference](#6-api-reference)
7. [Configuration Reference](#7-configuration-reference)
8. [Multi-Agent Protocols](#8-multi-agent-protocols)
9. [Security and Alignment Requirements](#9-security-and-alignment-requirements)
10. [Infrastructure Specification](#10-infrastructure-specification)
11. [Evaluation and Metrics](#11-evaluation-and-metrics)
12. [Phased Implementation Plan](#12-phased-implementation-plan)
13. [Dependency Matrix](#13-dependency-matrix)
14. [Glossary](#14-glossary)

---

## 1. Purpose and Scope

DreamEngine is a framework for **semi-offline generative simulation and memory consolidation** in autonomous AI agents. It enables agents to improve policy through imagined experience — without requiring additional real-world interactions — by learning a world model that can generate synthetic trajectories for actor-critic training.

### 1.1 Problem Statement

Standard online RL agents are sample-inefficient: every gradient step requires a live environment interaction. For agents operating in costly or dangerous environments (financial markets, physical robots, industrial systems), this bottleneck limits practical deployability. DreamEngine addresses this by:

- Decoupling experience *collection* from policy *optimization*
- Enabling up to 1024 synthetic training steps per real environment step
- Providing safe counterfactual evaluation without live environment risk
- Preserving long-term memory across tasks through tiered consolidation

### 1.2 Scope

This specification covers:

- All DreamEngine core components and their interfaces
- Data schemas and Pydantic contracts
- Kubernetes deployment topology and Helm chart structure
- Ray actor graph and resource allocation model
- Security controls and alignment constraints
- Evaluation metrics and acceptance criteria for each development phase

This specification does **not** cover:

- Domain-specific environment implementations (trading simulators, ROS2 integrations, etc.)
- LLM provider selection or fine-tuning procedures
- User-facing dashboard or visualization tooling (tracked separately)

---

## 2. System Overview

### 2.1 The Dreaming Loop

DreamEngine implements a five-stage closed loop:

```
┌─────────────────────────────────────────────────────────────────┐
│                         WAKE PHASE                              │
│  Environment ──► EnvRunner ──► Redis Streams ──► TimescaleDB   │
└─────────────────────────────────────────────────────────────────┘
                              │
                    [Scheduler Trigger]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DREAM PHASE                              │
│                                                                 │
│  1. PERCEIVE   ──► Episode ingest, embedding, indexing         │
│  2. CONSOLIDATE ──► Replay buffer, knowledge graph, vector DB  │
│  3. IMAGINE    ──► World model rollouts, LLM scenarios         │
│  4. EVALUATE   ──► Fidelity scoring, constraint validation     │
│  5. INTEGRATE  ──► Staged policy update, EWC, rollback gate    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                    [Policy Feedback Loop]
                              │
                              ▼
                     Active Policy (πθ)
```

### 2.2 Architectural Layers

| Layer | Latency SLA | Resource Class | Failure Impact |
|---|---|---|---|
| **Perception** | < 10ms | CPU/GPU inference | Immediate task failure |
| **Memory** | < 100ms | High-bandwidth storage | Degraded learning, stale policies |
| **Generation** | 1s – 10min | GPU-intensive (batch) | Reduced policy improvement |
| **Simulation** | 10ms – 1min | CPU/GPU by fidelity | Invalid dream validation |
| **Learning** | 1min – 1hr | GPU training | Policy stagnation |

### 2.3 Key Design Principles

1. **Causal isolation** — dream pods have no production egress; dreamed actions cannot affect live systems
2. **Provenance** — every artifact (experience, trajectory, policy delta) carries full lineage metadata
3. **Validation gates** — no policy update deploys without passing shadow/canary evaluation
4. **Distributed-first** — stateless workers where possible; actor-based stateful components with explicit persistence
5. **Pluggable backends** — memory, generation, and simulation layers are swappable via defined interfaces

---

## 3. Formal Definitions

### 3.1 Markov Decision Process

Let **M = (S, A, P, R, γ)** be a Markov Decision Process where:
- **S** — state space
- **A** — action space
- **P(s' | s, a)** — true transition dynamics
- **R(s, a)** — reward function
- **γ ∈ [0,1)** — discount factor

### 3.2 World Model

The agent maintains a world model **W_φ** parameterized by **φ** approximating true dynamics:

```
P̂(s' | s, a; φ) ≈ P(s' | s, a)
```

### 3.3 Dreaming as Semi-Offline Policy Optimization

The dreaming process generates imagined trajectories:

```
τ̃ = (s̃₀, ã₀, r̃₀, s̃₁, ..., s̃ₕ)
```

by sampling initial states from replay buffer **D** and rolling out **W_φ** with policy **πθ**.

Policy improvement combines real and dreamed data:

```
θ' = θ + α∇θ[J(πθ; D) + λJ(πθ; τ̃)]
```

where **λ** weights dream contribution, adaptively reduced when world model uncertainty is high.

### 3.4 RSSM Latent Dynamics

The Recurrent State-Space Model (RSSM) uses dual latent paths:

```
hₜ = f_GRU(hₜ₋₁, zₜ₋₁, aₜ₋₁)          # deterministic: predictable dynamics
zₜ ~ q(zₜ | hₜ, oₜ)                      # stochastic posterior (during training)
zₜ ~ p(zₜ | hₜ)                          # stochastic prior (during imagination)
```

### 3.5 Dream Quality Metrics

**Reward correlation:**
```
ρ_reward = Corr(R_dream, R_real)
```
Target: ρ > 0.7 for standard operation; < 0.5 triggers world model retraining.

**Distribution fidelity (MMD):**
```
MMD²(P_real, P_dream) = ‖E[φ(x)] − E[φ(y)]‖²_H
```
Computed using RBF kernel over encoded latent states; reported every dream cycle.

**Forgetting curve (memory relevance decay):**
```
R(t) = R₀ · exp(−λt / √(n_rehearsal + 1))
```
where λ is domain-specific decay rate and n_rehearsal counts dream-phase retrievals.

---

## 4. Data Contracts

All inter-component data exchange is defined via Pydantic v2 models. Protocol Buffer definitions exist for high-throughput paths (Redis streams, Ray object store).

### 4.1 ExperienceRecord

```python
class ExperienceRecord(BaseModel):
    obs: np.ndarray              # shape: (*obs_shape)
    action: np.ndarray           # shape: (*act_shape)
    reward: float
    next_obs: np.ndarray         # shape: (*obs_shape)
    done: bool
    info: dict[str, Any]
    timestamp: datetime
    agent_id: str
    episode_id: UUID
```

### 4.2 Episode

```python
class Episode(BaseModel):
    episode_id: UUID
    agent_id: str
    environment_version: str
    records: list[ExperienceRecord]
    total_return: float
    length: int
    success_flag: bool
    embedding: np.ndarray        # shape: (embed_dim,)  — episode summary vector
    dream_provenance: UUID | None  # null for real episodes
    metadata: dict[str, Any]
    start_time: datetime
    end_time: datetime
```

### 4.3 DreamTrajectory

```python
class DreamTrajectory(BaseModel):
    dream_id: UUID
    agent_id: str
    generation_mode: Literal["standard", "exploratory", "adversarial", "counterfactual"]
    fidelity_level: int          # 0=real, 1=latent, 2=decoded, 3=twin
    world_model_version: str
    latent_states: list[np.ndarray]
    actions: list[np.ndarray]
    imagined_rewards: list[float]
    predicted_values: list[float]
    uncertainty_estimates: list[float]
    source_episode_ids: list[UUID]
    validation_status: Literal["pending", "validated", "rejected"]
    quality_metrics: DreamQualityMetrics
    timestamp: datetime
```

### 4.4 DreamQualityMetrics

```python
class DreamQualityMetrics(BaseModel):
    predictive_consistency: float  # 0–1; inverse uncertainty
    value_stability: float         # 0–1; value range check
    action_diversity: float        # std dev of action distribution
    fidelity_score: float          # discriminator-based real/dream similarity
    constraint_satisfaction: float # fraction of symbolic constraints passed
    reward_correlation: float      # ρ(R_dream, R_real) on held-out data
```

### 4.5 PolicyUpdate

```python
class PolicyUpdate(BaseModel):
    update_id: UUID
    agent_id: str
    parameter_deltas: dict[str, np.ndarray]  # layer_name → delta tensor
    validation_scores: dict[str, float]
    dream_provenance: list[UUID]              # contributing dream_ids
    rollback_checkpoint: str                 # S3/GCS path to previous weights
    deployment_stage: Literal["shadow", "canary", "gradual", "full"]
    ewc_fisher: dict[str, np.ndarray] | None # Fisher information for EWC
    kl_from_wake: float                      # KL divergence from wake-phase policy
    created_at: datetime
```

### 4.6 MCP Dream Schema (JSON)

Standard schema for cross-agent dream exchange over MCP:

```json
{
  "dream_id": "uuid",
  "agent_id": "string",
  "timestamp": "ISO-8601",
  "dream_type": "exploration|consolidation|adversarial|counterfactual",
  "fidelity_level": 0,
  "world_model_version": "semver",
  "latent_trajectory": {
    "initial_state": "base64_tensor",
    "actions": ["base64_tensors"],
    "predicted_rewards": [0.0],
    "predicted_values": [0.0],
    "uncertainty_estimates": [0.0]
  },
  "decoded_observations": null,
  "validation_status": "pending|validated|rejected",
  "quality_metrics": {
    "predictive_accuracy": 0.0,
    "value_consistency": 0.0,
    "constraint_satisfaction": 0.0
  },
  "provenance": {
    "source_episodes": ["uuid"],
    "generative_model": "model_id",
    "random_seed": 0
  }
}
```

MCP resource URI format: `dream://agent_id/dream_id`

---

## 5. Component Specifications

### 5.1 Memory Substrate

#### 5.1.1 Tiered Storage Architecture

| Tier | Technology | Retention | Latency SLA | Write Target |
|---|---|---|---|---|
| **Hot** | Redis 7 cluster + NVMe SSD | 0–7 days | < 1ms | 100K msgs/sec |
| **Warm** | TimescaleDB hypertables + SATA SSD | 7–90 days | < 100ms | Async batch (15 min) |
| **Cold** | S3-compatible object storage | 90 days–1yr | < 1000ms | Lifecycle migration |
| **Archive** | Glacier / Deep Archive | 1+ years | Minutes–hours | Automated retention policy |

#### 5.1.2 Redis Data Structures

| Structure | Key Pattern | Use Case | Performance Target |
|---|---|---|---|
| Streams (`XADD/XREAD`) | `experience:{agent_id}` | Time-ordered episode ingest | 100K msgs/sec write |
| Sorted Sets (`ZADD`) | `episodes:priority:{agent_id}` | Priority-ranked retrieval | < 1ms query |
| Hashes | `agent:state:{agent_id}` | Agent state, scheduler params | < 1ms r/w |
| Pub/Sub | `dream:triggers` | Dream trigger broadcasting | < 1ms propagation |
| JSON (RedisJSON) | `overlay:{domain}:{version}` | Symbolic constraint overlays | < 5ms query |

Eviction policy: `allkeys-lfu` for experience embeddings; `volatile-lru` fallback. TTL default: 24h for experience streams; no TTL for pinned policy parameters.

#### 5.1.3 TimescaleDB Schema

```sql
CREATE TABLE episodes (
    episode_id          UUID PRIMARY KEY,
    agent_id            VARCHAR(64) NOT NULL,
    environment_version VARCHAR(32),
    start_time          TIMESTAMPTZ NOT NULL,
    end_time            TIMESTAMPTZ,
    total_reward        DOUBLE PRECISION,
    trajectory_length   INTEGER,
    dream_provenance    UUID REFERENCES dream_cycles(dream_id),
    metadata            JSONB,
    compressed_trajectory BYTEA  -- LZ4-compressed tensor sequence
);

SELECT create_hypertable('episodes', 'start_time',
    chunk_time_interval => INTERVAL '7 days');

CREATE INDEX idx_episodes_agent_time ON episodes(agent_id, start_time DESC);
CREATE INDEX idx_episodes_dream ON episodes(dream_provenance)
    WHERE dream_provenance IS NOT NULL;
```

Compression: `timescaledb.compress` enabled after 7-day chunk age; target 5–10× storage reduction.

Retention policy:
- Uncompressed chunks: 90 days
- Compressed chunks: 1 year
- Cold migration: automated via TimescaleDB job scheduler

#### 5.1.4 Qdrant Collection Configuration

```python
qdrant_client.create_collection(
    collection_name="dream_episodes",
    vectors_config=VectorParams(
        size=1024,          # episode summary embedding dimension
        distance=Distance.COSINE
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100,
        full_scan_threshold=10000
    ),
    optimizers_config=OptimizersConfigDiff(
        memmap_threshold=20000
    )
)
```

Collections required:
- `dream_episodes` — episode-level embeddings (dim: 1024)
- `dream_observations` — observation-level embeddings (dim: 256–1024)
- `dream_actions` — action embeddings (dim: 64–256)
- `shared_dreams` — cross-agent dream pool (dim: 1024)

#### 5.1.5 Memory Modality Summary

| Modality | Backend | Embedding Source | Typical Dimension | Primary Use |
|---|---|---|---|---|
| Episodic | TimescaleDB + EpisodeReplayBuffer | Episode summary encoder | 512–2048 | Dream seeding, verbatim replay |
| Semantic | Neo4j / Neptune | LLM extraction | Graph nodes/edges | Knowledge consolidation |
| Vector | Qdrant HNSW | World model encoder | 256–1024 | Similarity-based retrieval |
| Temporal | TimescaleDB hypertables | Raw time-series | N/A (time-indexed) | Causal chain traversal |

---

### 5.2 Dream Scheduler

#### 5.2.1 Trigger Mechanisms

The scheduler maintains four concurrent trigger channels, aggregated into a single priority queue:

**T1 — Idle Detection**
- Monitor: Redis TTL heartbeat expiry per agent
- Condition: GPU utilization < 15–20% OR CPU request queue depth < threshold
- Dream type: Micro-dream (H=5–15 rollout steps, single gradient update)
- Implementation: `redis.expire(heartbeat_key, TTL)` → pub/sub on expiry

**T2 — Entropy Trigger**
- Monitor: Running average of `H(π(·|s)) = -Σ_a π(a|s) log π(a|s)` tracked in Redis
- Condition: `H(π(·|s_t)) > τ_H · H̄_recent + σ_H` for ≥ 10–20% of a batch
- τ_H: 1.5–2.0 (configurable)
- Dream type: Targeted exploration dream

**T3 — Reward Surprise**
- Monitor: Per-step `|r_t − r̂_t| / √(Var(r̂) + ε)` streamed to Redis
- Condition: surprise > threshold (default: 2.0 normalized units)
- Dream type: Positive surprise → generalization dream; negative surprise → adversarial recovery dream
- Distributional shift: MMD or KS test comparing recent 1K steps vs. training baseline; trigger "adaptation dream" on detection

**T4 — Scheduled Deep Dream**
- Implementation: Kubernetes CronJob
- Default schedule: `0 2 * * *` (2 AM daily); configurable per agent profile
- Dream type: Full consolidation — replay buffer re-optimization, knowledge graph merge, world model retraining option, cross-agent pool merge

#### 5.2.2 Priority Queue

```
P0  CRITICAL    Safety violation, catastrophic forgetting     Immediate, preempt all
P1  HIGH        Entropy spike, reward surprise                 15-min SLA, preempt batch
P2  NORMAL      Idle opportunity                               1-hour SLA, best effort
P3  BACKGROUND  Scheduled consolidation                        24-hour SLA, spot instances
```

#### 5.2.3 Resource Arbitration

- Kubernetes `ResourceQuota` per namespace enforces GPU/CPU ceilings
- Ray scheduling primitives handle intra-cluster preemption
- Dream workers expose `yield` semantics: higher-priority wake tasks preempt in-progress low-priority dreams
- Pre-warming: scheduled P3 dreams trigger `scale-up` 15 minutes before execution

---

### 5.3 Generative Engine

#### 5.3.1 World Model Interface

All world model implementations must satisfy the following interface:

```python
class WorldModelBase(Protocol):
    def encode(self, obs: Tensor) -> Tensor:
        """Compress observation to latent representation. Shape: (*obs) → (latent_dim,)"""

    def step(
        self,
        latent: Tensor,
        action: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, dict]:
        """Latent transition. Returns: (next_latent, reward, done, info)"""

    def decode(self, latent: Tensor) -> Tensor:
        """Reconstruct observation from latent. Shape: (latent_dim,) → (*obs)"""

    def imagine(
        self,
        initial_latent: Tensor,
        policy: Callable,
        horizon: int
    ) -> DreamTrajectory:
        """Roll out imagined trajectory for H steps."""
```

#### 5.3.2 DreamerV3 RSSM Implementation

**Architecture parameters:**

| Parameter | XS | S | M | L | XL |
|---|---|---|---|---|---|
| Parameters | 1M | 10M | 100M | 300M | 1B+ |
| GRU hidden dim | 256 | 512 | 1024 | 2048 | 4096 |
| Categorical latents | 16×16 | 32×32 | 32×32 | 32×32 | 64×64 |
| MLP depth | 2 | 3 | 4 | 5 | 6 |

**Training objective:**

```
L_world = L_recon + L_reward + L_terminal + β · L_KL

L_KL = max(KL(q(z|h,o) ‖ p(z|h)), free_bits)
```

Key hyperparameters:
- Batch size B=16, sequence length T=64
- KL free bits: 1.0 nat (prevents posterior collapse)
- Symlog transformation: `sign(x) · ln(|x|+1)` on observations and rewards
- Gradient clipping: global norm 1000.0 (world model), implicit for actor/critic

**Actor training:**

```python
# Imagination horizon H=15 (default)
# λ-return with λ=0.95, γ=0.99
# Entropy regularization: entropy_scale=0.003
# Optimizer: Adam, lr=4e-5 (actor), lr=1e-4 (critic)
```

#### 5.3.3 LLM Backend (LiteLLM Router)

Used for symbolic and language-mediated domains. Dream prompt structure:

```
[SYSTEM: domain constraints + active symbolic overlays]
[MEMORY: top-k retrieved episodes as in-context examples]
[TASK: generation objective — continuation | counterfactual | adversarial]
[FORMAT: output schema for structured parsing]
```

Model routing strategy:

| Use Case | Primary Model | Fallback Chain |
|---|---|---|
| Counterfactual / causal | claude-3-opus | gpt-4-turbo → local vLLM |
| Long-context narrative | claude-3-5-sonnet | gpt-4o → local |
| High-volume, low-complexity | gpt-3.5-turbo | local vLLM |
| Privacy-sensitive / offline | local vLLM (mistral-7b) | None |

Budget management: per-cycle cost limit configurable; budget exhaustion routes to local model.

#### 5.3.4 Diffusion Backend

Latent diffusion model (LDM) conditioned on `(h_t, a_t)` sequence. Suitable for:
- Visual observation spaces (robotics, vision-based agents)
- Safety-critical domains requiring human-inspectable imagined states

Computational cost: only suitable for scheduled deep dreams, not opportunistic cycles.

#### 5.3.5 Fidelity Spectrum

| Level | Mechanism | Speed (steps/sec) | Use Case |
|---|---|---|---|
| **0 — Latent only** | Pure RSSM rollout, no decoding | 10,000+ | Exploration, policy optimization |
| **1 — Classifier-guided** | Latent rollout + outcome prediction | 5,000–8,000 | Safety filtering, rapid validation |
| **2 — Periodic decode** | Latent rollout + reconstruct every N steps | 1,000–5,000 | Validation, human inspection |
| **3 — Symbolic simulation** | Abstract state machine execution | 100–1,000 | Logical/relational domains |
| **4 — Full digital twin** | Complete environment clone | 1–100 | Final validation, safety-critical |

---

### 5.4 Simulation Environment

#### 5.4.1 Internal Sandbox

- Executes imagined trajectories entirely in learned latent space
- Target throughput: ≥ 10,000 step transitions/sec on A100 (batch=256, H=15)
- Latent-space safety classifiers reject trajectories entering dangerous regions
- Each `DreamWorker` Ray actor holds its own world model replica for parallel generation

#### 5.4.2 External Digital Twin

- Separate Kubernetes namespace; no network path to production
- State synchronization: production state feeds twin initialization on each dream cycle
- Supported backends: MuJoCo, Isaac Sim (robotics); historical replay engine (trading); custom domain simulators
- Data sync: bidirectional streaming — production state in, twin predictions out to validation metrics

#### 5.4.3 Symbolic Validators

Domain constraints stored as Datalog/Prolog facts in a symbolic database. Integrated via MCP constraint-satisfaction tool. Validation runs synchronously before policy integration.

Constraint categories:
- **Physical** — energy conservation, kinematic limits
- **Logical** — no-arbitrage, legal consistency
- **Regulatory** — compliance rules (GDPR, financial regulations)
- **Domain-specific** — custom business logic

---

### 5.5 Policy Update Loop

#### 5.5.1 Off-Policy Correction

Dreamed data is weighted by importance ratio:

```
w_t = π(a_t|s_t) / π_b(a_t|s_t)
```

For model-based dreaming where π_dream = π_target, importance weights are unity; correction applied for dreams generated from historical policy checkpoints.

#### 5.5.2 Elastic Weight Consolidation (EWC)

Protects critical parameters during dream updates:

```
L_EWC(θ) = L_dream(θ) + (λ/2) · Σᵢ Fᵢ(θᵢ − θ*ᵢ)²
```

where Fᵢ = diagonal Fisher information (computed over wake-phase data), θ* = post-wake policy checkpoint.

#### 5.5.3 KL Divergence Constraint

Limits policy deviation from wake-phase baseline:

```
L_reg(π) = L_RL(π) − β · D_KL(π ‖ π_wake)
```

β schedule: adaptive — increased when dream fidelity is low, reduced as validation improves.

#### 5.5.4 Staged Rollout Gate

| Stage | Traffic | Duration | Exit Criteria | Rollback Trigger |
|---|---|---|---|---|
| **Shadow** | 0% (parallel) | 1–24 hours | Performance ≥ baseline on real data | N/A |
| **Canary** | 1–10% | 1–7 days | Safety constraints satisfied; no anomaly alerts | Any constraint violation |
| **Gradual** | 10–50% | 3–14 days | Statistically significant improvement on key metrics | Performance degradation > 5% |
| **Full** | 100% | Ongoing | Continuous monitoring | Automatic rollback on anomaly |

Rollback: atomic policy version swap via Kubernetes blue-green deployment. Policy history retained for ≥ 10 versions.

---

### 5.6 Orchestration Layer

#### 5.6.1 Ray Actor Graph

```
DreamOrchestrator (single process, lightweight scheduler)
    │
    ├── EnvRunner actors (n=4–50)
    │     Real or simulated environment interaction
    │     Resource: 2 CPU, 0.5 GPU each
    │
    ├── DreamWorker actors (n=2–100)
    │     World model replicas, trajectory generation
    │     Resource: 4 CPU, 1 GPU each (standard)
    │
    └── LearnerGroup (n=1–8)
          Policy and world model gradient updates
          Resource: 8–16 CPU, 1–4 GPU each
```

#### 5.6.2 DreamWorker Actor Specification

```python
@ray.remote(num_gpus=1, num_cpus=4, memory=32*1024**3)
class DreamWorker:
    def generate_dream_rollouts(
        self,
        seed_episodes: list[EpisodeBatch],
        generation_mode: str = "standard"
    ) -> DreamBatch: ...

    def evaluate_dream_quality(
        self,
        dream_batch: DreamBatch
    ) -> dict[str, float]: ...

    def get_stats(self) -> dict: ...
```

See `dreamengine/orchestration/dream_worker.py` for full implementation.

#### 5.6.3 Shared Replay Buffer (Zero-Copy)

Episode objects stored in Ray object store (Plasma / shared memory). EnvRunner and Learner actors exchange data via object references — no serialization overhead for same-node access; Apache Arrow serialization for cross-node transfers.

```
[EnvRunner actors] ──(object refs)──► [Ray Object Store] ◄──(object refs)── [Learner actors]
                                              │
                                     [Plasma / Shared Memory]
                                              │
                                     [Zero-copy deserialization]
```

---

## 6. API Reference

### 6.1 DreamEngine Core API

#### `dreamengine.run(config: DreamEngineConfig) → DreamEngineHandle`

Initialize the DreamEngine runtime. Connects to Redis, TimescaleDB, Qdrant; initializes Ray cluster; starts scheduler.

#### `DreamEngineHandle.register_agent(agent_id: str, env_config: EnvConfig, world_model_config: WorldModelConfig) → AgentHandle`

Register a new agent with the engine. Creates per-agent memory partitions and scheduler entry.

#### `AgentHandle.step(obs: np.ndarray, action: np.ndarray, reward: float, done: bool, info: dict) → None`

Ingest one environment transition. Writes to Redis stream; triggers scheduler evaluation asynchronously.

#### `AgentHandle.act(obs: np.ndarray) → np.ndarray`

Query the agent's active policy. Latency SLA: < 10ms.

#### `AgentHandle.get_policy_version() → str`

Returns the currently deployed policy version identifier.

#### `AgentHandle.trigger_dream(mode: str = "standard", priority: int = 2) → DreamHandle`

Manually trigger a dream cycle. Returns a handle to monitor progress.

#### `DreamHandle.status() → DreamStatus`

Poll dream cycle progress. Returns `{stage, percent_complete, quality_metrics}`.

#### `DreamHandle.await_completion(timeout: float = 600.0) → DreamResult`

Block until dream cycle completes or timeout. Returns final quality metrics and policy update status.

### 6.2 Memory API

#### `dreamengine.memory.retrieve(agent_id: str, query: RetrievalQuery, k: int = 16) → list[Episode]`

Hybrid retrieval across episodic, vector, and temporal modalities. Applies context window optimization scoring (recency 0.15, relevance 0.35, surprise 0.25, diversity 0.15, quality 0.10).

#### `dreamengine.memory.consolidate(agent_id: str) → ConsolidationResult`

Trigger manual semantic consolidation. Extracts entities and relations from recent episodes into the knowledge graph.

### 6.3 Scheduler API

#### `dreamengine.scheduler.set_profile(agent_id: str, profile: SchedulerProfile) → None`

Set scheduler parameters: entropy threshold multiplier, surprise threshold, deep-dream schedule, dream mode, training ratio.

#### `dreamengine.scheduler.get_stats(agent_id: str) → SchedulerStats`

Returns current trigger state, queue depth, last dream timestamp, and dream utility history.

---

## 7. Configuration Reference

### 7.1 DreamConfig

```python
@dataclass
class DreamConfig:
    # World model
    world_model_size: Literal["xs", "s", "m", "l", "xl"] = "s"
    imagination_horizon: int = 15          # H — dream rollout steps
    training_ratio: int = 1024             # dream steps per real step
    batch_size: int = 16                   # B
    sequence_length: int = 64             # T

    # Generation
    fidelity_level: int = 1               # 0–4
    temperature: float = 1.0              # latent sampling temperature
    num_dream_sequences: int = 16

    # Scheduler
    idle_gpu_threshold: float = 0.20      # GPU utilization below this triggers idle dreams
    entropy_threshold_multiplier: float = 1.5
    surprise_threshold: float = 2.0       # normalized surprise units
    deep_dream_schedule: str = "0 2 * * *"  # cron expression

    # Policy update
    lambda_dream: float = 1.0             # λ weighting dream contribution
    kl_beta: float = 0.1                  # KL constraint weight
    ewc_lambda: float = 1000.0            # EWC regularization strength
    rollback_threshold: float = 0.05      # 5% performance degradation

    # Infrastructure
    device: str = "cuda"
    ray_address: str | None = None        # None = local Ray cluster
    redis_url: str = "redis://localhost:6379"
    timescaledb_url: str = "postgresql://localhost:5432/dreamengine"
    qdrant_url: str = "http://localhost:6333"
```

### 7.2 SchedulerProfile Presets

| Profile | Use Case | Training Ratio | Deep Dream | Fidelity |
|---|---|---|---|---|
| `fast_iteration` | Research / rapid prototyping | 256 | Every 4h | Latent only |
| `standard` | General production | 1024 | Nightly | Mixed |
| `conservative` | Safety-critical / high-stakes | 128 | Nightly + human gate | Full digital twin |
| `edge` | Resource-constrained deployment | 64 | Weekly | Latent only |

---

## 8. Multi-Agent Protocols

### 8.1 Dream Sharing Models

| Model | Privacy | Configuration |
|---|---|---|
| `private` | Complete isolation | Default; no shared collections in Qdrant |
| `shared_pool` | Reputation-weighted access | Shared `shared_dreams` Qdrant collection with quality filtering |
| `hierarchical` | Tactical private / strategic shared | Multi-level Qdrant collections per hierarchy level |
| `federated` | (ε, δ)-DP | Gradient clipping + Gaussian noise before aggregation |

### 8.2 Shared Dream Pool

```python
class SharedDreamPool:
    def contribute_dream(
        self,
        agent_id: str,
        dream_trajectory: DreamTrajectory,
        metadata: dict
    ) -> str: ...  # returns point_id in Qdrant

    def retrieve_relevant_dreams(
        self,
        query_context: DreamTrajectory,
        k: int = 10,
        domain_filter: str | None = None,
        min_quality: float = 0.7
    ) -> list[ScoredDream]: ...
```

Retrieval ranking combines: (1) cosine similarity to query embedding, (2) contributor reputation score, (3) recency, (4) diversity from already-retrieved set.

### 8.3 MCP Tool Definitions

DreamEngine exposes the following MCP tools:

| Tool | Description | Input | Output |
|---|---|---|---|
| `generate_counterfactual` | Generate "what if" scenario for a past decision | `seed_episode_id`, `intervention_spec` | `DreamTrajectory` |
| `validate_dream` | Run validation pipeline on a trajectory | `dream_id` | `DreamQualityMetrics` |
| `consolidate_memory` | Trigger memory consolidation for an agent | `agent_id` | `ConsolidationResult` |
| `evaluate_policy` | Evaluate current policy against dreamed scenarios | `agent_id`, `scenario_config` | `PolicyEvalResult` |
| `get_dream_status` | Poll active dream cycle progress | `dream_id` | `DreamStatus` |

MCP resource URIs: `dream://agent_id/dream_id`

### 8.4 Hermes–OpenClaw Responsibility Split

| Concern | Hermes (Supervisor) | OpenClaw / NemoClaw (Worker) |
|---|---|---|
| Scheduler policy | ✓ Sets triggers, priorities, profiles | — |
| Dream admission | ✓ Decides which dreams to run | — |
| Validation decisions | ✓ Accept/reject policy updates | — |
| Rollout gate control | ✓ Controls deployment stages | — |
| Memory ingest jobs | — | ✓ Stateless write to Redis/TSDB |
| Latent rollout generation | — | ✓ DreamWorker execution |
| Validator runs | — | ✓ Fidelity scoring, constraint checks |
| Deploy tasks | — | ✓ Executes Helm/k8s deploy calls |

Workers are stateless; all state and provenance are maintained by Hermes through structured MCP/ACP messages.

---

## 9. Security and Alignment Requirements

### 9.1 Mandatory Controls

The following controls are **non-negotiable** and must be enforced at the infrastructure level, not application level:

1. **Network isolation** — `dreamengine-dream` namespace has zero egress to production, external APIs, or databases. Enforced via Kubernetes `NetworkPolicy`.
2. **Provenance on every artifact** — all `ExperienceRecord`, `DreamTrajectory`, and `PolicyUpdate` objects carry `agent_id`, `dream_id` (null for real experience), `world_model_version`, and `fidelity_level`.
3. **No auto-deploy** — policy updates are never applied without passing at least the shadow stage. The shadow→canary transition requires an explicit approval signal (automated metrics gate OR human approval).
4. **Rollback availability** — last 10 policy versions are retained and deployable within 60 seconds.
5. **Audit log** — all dream cycles, validation decisions, human approvals, and policy deployments are written to append-only audit log in TimescaleDB.

### 9.2 Risk Register

| Risk | Severity | Likelihood | Primary Mitigation | Secondary Mitigation |
|---|---|---|---|---|
| Confirmation bias loop | High | High | Adversarial dream generation; explicit failure replay | Diversity objectives in episode selection |
| Reality drift | High | Medium | MMD monitoring on every cycle; held-out real evaluation | World model retraining trigger on drift > threshold |
| Epistemic isolation | Medium | Low | Provenance tags + importance discount for dream data | Decision frameworks account for source reliability |
| Dream pool poisoning | High | Low | Byzantine-resilient aggregation (trimmed mean) | Reputation scoring; rate limiting; quality floor |
| Reward hacking | High | Medium | Transfer validation A/B test before canary | Conservative Q-learning (CQL) pessimism |
| Alignment drift | Critical | Low | Value-embedding constraints in generation prompt | Symbolic validators; multi-objective reward |
| Latent adversarial examples | Medium | Low | Adversarial training; input plausibility checks | Stochastic decoding averaging |
| Model extraction via query patterns | Low | Low | Rate limiting; differential privacy on outputs | Output perturbation |

### 9.3 Hallucination Drift Detection

Continuous monitoring required every dream cycle:

| Test | Applied To | Threshold |
|---|---|---|
| MMD (RBF kernel) | Latent state distributions | < 0.05 (trigger retraining > 0.1) |
| Reward correlation | ρ(R_dream, R_real) | > 0.7 (disable dreaming < 0.5) |
| Discriminator accuracy | Real vs. dreamed observations | < 60% accurate = high fidelity |
| Ensemble disagreement | Reward prediction variance | Flag if variance > 2× historical mean |

### 9.4 Human Oversight Hooks

| Trigger | Latency | Required Action |
|---|---|---|
| Novel dream type requested | Minutes | Approve generation parameters |
| Validation failure / anomaly alert | Hours | Inspect dream, override or confirm rejection |
| Policy update > 15% KL from wake baseline | Hours | Review diff, approve or block |
| Safety constraint violation detected | Seconds | Automatic halt + human notification |
| P0 critical escalation | Seconds | Emergency stop capability must be available |

---

## 10. Infrastructure Specification

### 10.1 Kubernetes Namespace Architecture

```
dreamengine-wake
  ├── Network: egress to production services only
  ├── Resources: CPU 100, Memory 500Gi
  └── Workloads: EnvRunner pods, policy inference

dreamengine-dream
  ├── Network: NO egress (internal only, strict isolation)
  ├── Resources: GPU 50, CPU 500, Memory 2Ti
  └── Workloads: DreamWorker pods, LearnerGroup pods

dreamengine-validate
  ├── Network: egress to staging environment only
  ├── Resources: CPU 50, Memory 200Gi
  └── Workloads: Shadow/canary policy evaluation

dreamengine-shared
  ├── Network: limited egress (monitoring, config)
  ├── Resources: CPU 20, Memory 100Gi
  └── Workloads: Redis, TimescaleDB, Qdrant, Prometheus, Grafana
```

### 10.2 Helm Chart Structure

```
k8s/helm/dreamengine/
├── Chart.yaml
├── values/
│   ├── values-production.yaml
│   ├── values-staging.yaml
│   └── values-edge.yaml
├── templates/
│   ├── memory/
│   │   ├── redis-cluster.yaml
│   │   ├── timescaledb.yaml
│   │   └── qdrant.yaml
│   ├── scheduler/
│   │   ├── deployment.yaml
│   │   ├── hpa.yaml
│   │   └── cronjob.yaml
│   ├── generation/
│   │   ├── dream-worker-gpu.yaml
│   │   └── vllm-service.yaml
│   ├── simulation/
│   │   ├── env-runner.yaml
│   │   └── digital-twin.yaml
│   └── learning/
│       ├── learner-group.yaml
│       └── tensorboard.yaml
└── charts/
    ├── prometheus/
    ├── grafana/
    └── jaeger/
```

### 10.3 Autoscaling Policy

| Component | Scale Metric | Scale Up | Scale Down | Min / Max |
|---|---|---|---|---|
| DreamWorker | GPU utilization > 70% | +50% replicas | −25% after 10min idle | 2 / 100 |
| EnvRunner | Queue depth > 100 | +2/min | −1 after 5min idle | 4 / 50 |
| LearnerGroup | Batch wait time > 30s | +1 replica | −1 after 15min idle | 1 / 8 |
| Memory (TimescaleDB) | Memory usage > 80% | +1 node | Manual | 3 / 10 |

### 10.4 Deep Dream CronJob Template

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: deep-dream-consolidation
  namespace: dreamengine-dream
spec:
  schedule: "0 2 * * *"
  concurrencyPolicy: Forbid
  jobTemplate:
    spec:
      activeDeadlineSeconds: 21600  # 6-hour timeout
      ttlSecondsAfterFinished: 86400
      template:
        spec:
          priorityClassName: dream-background
          containers:
          - name: dream-engine
            image: dreamengine/deep-dream:latest
            resources:
              requests:
                nvidia.com/gpu: 8
                memory: "512Gi"
                cpu: "64"
            env:
            - name: DREAM_MODE
              value: "deep_consolidation"
            - name: MAX_DURATION_MINUTES
              value: "360"
          restartPolicy: OnFailure
```

---

## 11. Evaluation and Metrics

### 11.1 Phase Acceptance Criteria

**Phase 1 (MVP):**
- [ ] Episode ingest → Redis → TimescaleDB pipeline operational; write latency < 1ms hot path
- [ ] Qdrant indexing within 60s of episode completion
- [ ] RSSM world model trains to convergence on target domain within 24h on single GPU
- [ ] Latent rollout throughput ≥ 1,000 steps/sec on available hardware
- [ ] Offline reward correlation ρ ≥ 0.7 on held-out episodes
- [ ] All data contracts (ExperienceRecord, Episode, DreamTrajectory, PolicyUpdate) validated end-to-end

**Phase 2 (Controlled Dreaming):**
- [ ] All four scheduler triggers firing correctly in integration tests
- [ ] Fidelity discriminator trained; accuracy ≥ 70% on real vs. dreamed samples
- [ ] KL constraint enforced; policy KL from wake baseline < 0.5 nats after dream cycle
- [ ] Shadow stage gate functional; canary deployment only after shadow pass
- [ ] Rollback completes in < 60 seconds

**Phase 3 (High-Fidelity Validation):**
- [ ] Digital twin synchronized with production state within 5-minute staleness tolerance
- [ ] Symbolic validator passes/rejects trajectories correctly on domain test suite
- [ ] End-to-end dream cycle (trigger → rollout → validate → deploy) operational in staging

**Phase 4 (Multi-Agent):**
- [ ] MCP dream schema validates against spec; exchange between two heterogeneous agents demonstrated
- [ ] Shared dream pool retrieval latency < 200ms for k=16
- [ ] Federated aggregation preserves (ε=1.0, δ=1e-5)-DP guarantee

### 11.2 Ongoing Performance Targets

| Metric | Target | Alert Threshold |
|---|---|---|
| Hot-path write latency | < 1ms p99 | > 5ms |
| Episode retrieval (Qdrant) | < 200ms p99 | > 500ms |
| Latent rollout throughput | ≥ 1,000 steps/sec | < 500 steps/sec |
| Reward correlation ρ | > 0.7 | < 0.5 (disable dreaming) |
| MMD (latent distribution) | < 0.05 | > 0.1 (trigger retraining) |
| Policy improvement per cycle | > 0% average | < −2% (rollback) |
| Wake-phase task success | Monotonically non-decreasing | Any 5% regression |

---

## 12. Phased Implementation Plan

### Phase 1 — Single-Agent MVP

**Goal:** Closed-loop single agent: ingest → store → train world model → generate latent rollouts → offline evaluation only. No automatic policy deployment.

**Scope:**
- `dreamengine/memory/` — Redis ingest pipeline, TimescaleDB storage, Qdrant indexing
- `dreamengine/schemas/` — All Pydantic data contracts
- `dreamengine/generation/world_model.py` — RSSM implementation (start with model size S)
- `dreamengine/orchestration/dream_worker.py` — `DreamWorker` Ray actor, `DreamConfig`
- `dreamengine/learning/learner.py` — Actor and critic training in imagination (no auto-deploy)
- Offline evaluation notebook: reward correlation, MMD, policy improvement vs. baseline

**Out of scope for Phase 1:** Scheduler, validation gate, LLM generation, diffusion, multi-agent.

**Target duration:** 6–8 weeks

---

### Phase 2 — Controlled Dreaming

**Goal:** All four scheduler triggers + full validation pipeline + staged rollout gate with rollback.

**Scope:**
- `dreamengine/scheduler/` — Idle, entropy, surprise, and cron triggers; priority queue
- `dreamengine/validation/` — Fidelity discriminator, uncertainty quantification, constraint checking
- `dreamengine/learning/gate.py` — Shadow/canary/gradual/full staged deployment
- `dreamengine/learning/ewc.py` — Elastic Weight Consolidation
- Kubernetes CronJob for deep dreams
- Rollback mechanism (blue-green policy versioning)
- Prometheus metrics + Grafana dashboard for dream quality

**Target duration:** 6–8 weeks

---

### Phase 3 — High-Fidelity Validation

**Goal:** Digital twin + symbolic validators + domain-specific adapters.

**Scope:**
- `dreamengine/simulation/digital_twin.py` — Digital twin state sync
- `dreamengine/simulation/symbolic_validator.py` — Datalog/Prolog constraint engine via MCP
- LiteLLM router integration for LLM-based dream generation
- LangGraph orchestration for hybrid multi-model generation
- Domain adapter interfaces (trading, robotics, mining)

**Target duration:** 8–10 weeks

---

### Phase 4 — Multi-Agent Orchestration

**Goal:** Cross-agent dream sharing, MCP schema, federated privacy, Hermes integration.

**Scope:**
- `dreamengine/multiagent/shared_pool.py` — Shared Qdrant dream pool
- `dreamengine/multiagent/federated.py` — DP-SGD gradient aggregation
- `dreamengine/mcp/tools.py` — MCP tool definitions (generate_counterfactual, validate_dream, etc.)
- `dreamengine/mcp/schemas.py` — MCP dream schema JSON serialization
- Hermes–OpenClaw integration layer

**Target duration:** 8–10 weeks

---

## 13. Dependency Matrix

| Package | Version | Layer | Required By |
|---|---|---|---|
| `ray[default]` | ≥ 2.9 | Orchestration | DreamWorker, EnvRunner, LearnerGroup |
| `ray[rllib]` | ≥ 2.9 | Learning | EpisodeReplayBuffer, LearnerGroup |
| `torch` | ≥ 2.2 | Generation, Learning | RSSM, actor, critic |
| `redis` | ≥ 5.0 | Memory | Hot tier, scheduler |
| `psycopg2` | ≥ 2.9 | Memory | TimescaleDB connector |
| `qdrant-client` | ≥ 1.8 | Memory | Vector search |
| `pydantic` | ≥ 2.0 | Schemas | All data contracts |
| `litellm` | ≥ 1.30 | Generation | LLM routing |
| `langgraph` | ≥ 0.1 | Generation | Hybrid orchestration |
| `prefect` | ≥ 2.14 | Orchestration | Pipeline workflow |
| `prometheus-client` | ≥ 0.20 | Observability | Metrics export |
| `numpy` | ≥ 1.26 | All | Tensor utilities |
| `pyarrow` | ≥ 15.0 | Memory | Ray object store serialization |

---

## 14. Glossary

| Term | Definition |
|---|---|
| **Dream** | A synthetic experience trajectory generated by the world model in latent space, not involving real environment interaction |
| **DreamWorker** | A stateless Ray actor that holds a world model replica and generates dream trajectory batches |
| **Dreaming Loop** | The five-stage cycle: Perceive → Consolidate → Imagine → Evaluate → Integrate |
| **EWC** | Elastic Weight Consolidation; a regularization technique that prevents catastrophic forgetting by penalizing changes to parameters deemed critical by the Fisher information matrix |
| **Fidelity Level** | An integer 0–4 indicating simulation accuracy: 0=latent only, 1=classifier-guided, 2=periodic decode, 3=symbolic, 4=full digital twin |
| **KL Constraint** | A bound on the KL divergence between the updated dream-trained policy and the wake-phase policy baseline, preventing excessive policy drift |
| **MMD** | Maximum Mean Discrepancy; a kernel-based statistical test used to measure distribution distance between real and dreamed state distributions |
| **Policy Provenance** | Lineage metadata on a policy update identifying which dream cycles, world model version, and episodes contributed to it |
| **RSSM** | Recurrent State-Space Model; the DreamerV3 world model architecture with dual deterministic/stochastic latent paths |
| **Reality Gap** | The divergence between dreamed and real dynamics; addressed via multi-fidelity simulation, validation gates, and conservative value estimation |
| **Reality Drift** | Progressive compounding of errors over iterative dream-policy cycles, causing world model dynamics to diverge from reality |
| **Symbolic Overlay** | Declarative domain constraints stored as Datalog/Prolog facts that dreamed trajectories must satisfy before policy integration |
| **Training Ratio** | The number of dream steps taken per real environment step; default 1024:1 |
| **Wake Phase** | The period of real environment interaction, real-time inference, and experience ingest |
| **Dream Phase** | The offline period of memory consolidation, world model imagination, and policy optimization |
| **Validation Gate** | The staged rollout pipeline (shadow → canary → gradual → full) that every policy update must pass before production deployment |
