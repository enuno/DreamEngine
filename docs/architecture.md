# DreamEngine Architecture

DreamEngine is a distributed architecture for generative simulation, memory consolidation, and policy refinement in autonomous AI agents. It separates wake-phase operation from dream-phase computation so agents can collect real experience online while performing semi-offline imagination, counterfactual analysis, and policy improvement in isolated runtime environments.[file:1]

## Architectural principles

The system is organized around five functional layers: perception, memory, generation, simulation, and learning. This separation exists because each layer has different latency constraints, resource profiles, scaling drivers, and failure modes, and therefore should not be tightly coupled in one runtime path.[file:1]

DreamEngine is distributed-first by design. Stateless work should be scheduled as short-lived tasks, while stateful components such as environment runners, learners, and dream workers should run as actors with explicit persistence and checkpoint boundaries.[file:1]

Dream computation must remain causally isolated from production systems. Dream namespaces, pods, and workers may read from controlled memory substrates and validation environments, but they must not be able to trigger external side effects in production APIs, financial systems, or physical infrastructure.[file:1]

## System layers

| Layer | Role | Latency profile | Primary technologies |
|---|---|---:|---|
| Perception | Real-time sensing, acting, and reward collection | Sub-10 ms to low milliseconds | EnvRunner actors, feature encoders, Redis streams [file:1] |
| Memory | Episodic, semantic, vector, and temporal persistence | Sub-100 ms retrieval on hot/warm paths | Redis, TimescaleDB, Qdrant, knowledge graph store [file:1] |
| Generation | Produces imagined or counterfactual trajectories | Seconds to minutes | RSSM / DreamerV3, LiteLLM, diffusion backends [file:1] |
| Simulation | Executes dream scenarios at selected fidelity | Milliseconds to minutes | Latent sandbox, symbolic simulators, digital twins [file:1] |
| Learning | Refines world model and policy from real and dreamed data | Minutes to hours | LearnerGroup actors, distributed GPU training, rollout gates [file:1] |

The default execution model is wake online, dream offline. Wake execution prioritizes responsiveness and stable interaction with the environment, while dream execution prioritizes batch throughput, exploration, and system maintenance.[file:1]

## Dreaming loop

DreamEngine operates as a closed loop:

1. **Perceive**: collect environment transitions, predictions, rewards, and policy metadata.[file:1]
2. **Consolidate**: store trajectories, update indexes, extract semantic structure, and compute retrieval features.[file:1]
3. **Imagine**: generate trajectories through latent rollouts, LLM scenario construction, or hybrid pipelines.[file:1]
4. **Evaluate**: score fidelity, uncertainty, constraint satisfaction, and expected policy value.[file:1]
5. **Integrate**: update the world model and candidate policy, then stage rollout only after validation gates pass.[file:1]

This loop is adaptive. Dream utility metrics feed back into scheduler policy so the platform can increase dream frequency or fidelity when dreaming is productive and reduce it when real data collection is more valuable.[file:1]

## Memory subsystem

DreamEngine uses a tiered, multi-modal memory substrate because no single database is a good fit for all access patterns. Recent experience requires fast append and retrieval, historical trajectories require efficient temporal storage, and dream seeding requires semantic nearest-neighbor search across large corpora.[file:1]

### Episodic memory

Episodic memory stores complete trajectories rather than independent transitions. The paper explicitly favors episode-level storage because preserving temporal and causal structure is important for world model training and counterfactual replay.[file:1]

A typical episode record includes:

- Ordered observation, action, reward, and termination sequences.[file:1]
- Episode metadata such as return, length, success flag, timestamps, and agent identity.[file:1]
- Optional embeddings and dream provenance for downstream retrieval and validation.[file:1]

TimescaleDB hypertables are used as the warm trajectory archive, while Redis streams act as the hot ingestion layer. Recent data lands in Redis first for sub-millisecond writes and immediate consumption, then is batch-flushed into TimescaleDB for durable temporal storage and aggregation.[file:1]

### Semantic memory

Semantic memory is produced by consolidation. DreamEngine extracts entities, relations, and abstractions from trajectories and stores them in a property graph or symbolic substrate so that the system can reason over patterns that are not obvious from raw time-series alone.[file:1]

Semantic memory is also where domain rules live. Constraint overlays such as market rules, robotics limits, or procedural/legal restrictions can be attached to the dream pipeline and used during evaluation to reject or repair invalid dreamed trajectories before integration.[file:1]

### Vector memory

Vector memory provides similarity-based retrieval across observations, actions, episodes, and tasks. Qdrant is the preferred vector store in the paper because it supports dense vector search with sparse metadata filtering and distributed operation.[file:1]

Embeddings are produced at multiple granularities so retrieval can operate at the right abstraction level:

| Embedding level | Typical use |
|---|---|
| Observation | Fine-grained state similarity and local recall [file:1] |
| Action | Behavioral similarity and policy pattern matching [file:1] |
| Episode | Dream seeding from situationally similar trajectories [file:1] |
| Task | Cross-task transfer and long-range retrieval [file:1] |

### Temporal memory

Temporal memory stores state evolution with explicit time indexing and optional inferred causal links. This enables not only replay but also root-cause tracing, failure-path retrieval, and sequence-aware counterfactual construction.[file:1]

### Memory lifecycle

The storage lifecycle is hot, warm, cold, and archive. Redis covers the hot tier, TimescaleDB covers warm history, and object storage or archival storage handles longer retention windows. Migration policy should consider access frequency, age, and measured computational utility rather than time alone.[file:1]

## Retrieval and dream seeding

Dream quality depends heavily on context selection. DreamEngine does not treat dream generation as unconditional sampling; instead it uses retrieval-augmented generation across the memory substrate so that simulated futures remain relevant to the current state, task, and uncertainty profile.[file:1]

Episode selection should balance:

- Recency, to account for current environment regime.[file:1]
- Relevance, to match the current embedding context.[file:1]
- Surprise, to emphasize informative failures and unexpected rewards.[file:1]
- Diversity, to avoid narrow replay of near-duplicates.[file:1]
- Quality, to prefer previously validated or productive memories.[file:1]

Cross-modal retrieval is a first-class requirement. Queries should be able to combine symbolic constraints, vector similarity, and temporal windows in one retrieval plan, then merge candidates with a learned or rules-based re-ranking stage.[file:1]

## Dream scheduler

The scheduler is the control plane for dream execution. Its job is not only to launch batch jobs but also to decide *when* to dream, *how deeply* to dream, and *which fidelity* to spend on each objective.[file:1]

### Trigger classes

DreamEngine defines four primary trigger families:

| Trigger type | Purpose | Example action |
|---|---|---|
| Idle-cycle trigger | Opportunistic micro-dreaming during low utilization | Short horizon replay and one-step updates [file:1] |
| Entropy trigger | Respond to high policy uncertainty | Bias sampling toward uncertain states [file:1] |
| Surprise/shift trigger | Respond to reward prediction error or distribution change | Launch adaptation or adversarial dreams [file:1] |
| Scheduled trigger | Perform expensive consolidation and retraining | Nightly or periodic deep dream jobs [file:1] |

Idle-cycle dreams should be cheap and preemptible. Scheduled deep dreams can be far more expensive and may include full replay buffer re-optimization, semantic consolidation, and world model retraining.[file:1]

### Prioritization

The scheduler should implement preemptive priority classes so high-risk or high-value dream requests can interrupt background maintenance. This maps naturally to Redis-backed priority queues plus Kubernetes quota and Ray scheduling controls.[file:1]

## Generative engine

The generative engine is intentionally pluggable, but the implementation path should be staged. The architecture supports multiple backends because different domains require different imagination modalities, yet the first production path should center on an RSSM world model with one clear rollout interface.[file:1]

### World model backend

The core backend is a DreamerV3-style recurrent state-space model. It maintains deterministic recurrent state for temporal coherence and stochastic latent state for uncertainty, which together support efficient long-horizon imagination in latent space.[file:1]

The minimal world model interface should expose:

```python
class WorldModel(Protocol):
    def encode(self, obs):
        ...

    def step(self, latent, action):
        ...

    def decode(self, latent):
        ...

    def imagine(self, initial_latent, policy, horizon):
        ...
```

That interface is sufficient to support training, latent rollout, decoded validation, and replacement with alternative backends such as transformer world models or graph-structured models later.[file:1]

### Latent-space sampling

The paper proposes several dream construction mechanisms that should be captured explicitly in the architecture:

- VAE-style latent navigation for controllable variation.[file:1]
- Mixture sampling across multiple experience embeddings to synthesize plausible composites.[file:1]
- Spherical interpolation for smooth state transitions and difficulty progression.[file:1]
- Constraint-aware sampling to keep generated states physically and logically plausible.[file:1]

These operations should be modeled as generation strategies rather than hard-coded into one dream worker. A clean interface lets the scheduler request `exploratory`, `counterfactual`, `adversarial`, or `consolidation` dreams with different sampling controls.[file:1]

### LLM and diffusion backends

LLM-based generation is appropriate when the domain is symbolic, language-mediated, or causally rich but not easily represented as low-dimensional state tensors. Diffusion-based generation is appropriate when observation-level realism matters, such as robotics vision, spectrograms, or industrial sensor arrays.[file:1]

Hybrid generation is a long-term capability, not the first milestone. In that mode, a coordinator decomposes one dream request into multiple modality-specific sub-tasks, then reconciles them through consistency checks and validation gates.[file:1]

## Dream generation modes

DreamEngine should support at least four explicit generation modes at the architecture level:

| Mode | Goal | Typical use |
|---|---|---|
| Standard | Extend likely futures from retrieved context | Planning and policy optimization [file:1] |
| Exploratory | Increase diversity through higher-temperature latent sampling | Coverage and novelty search [file:1] |
| Counterfactual | Intervene on prior actions or conditions | Regret analysis and causal reasoning [file:1] |
| Adversarial | Generate challenging but plausible failure cases | Robustness and safety testing [file:1] |

Counterfactual generation should be represented as an explicit intervention workflow rather than a generic sample mutation. The paper maps this to a Pearl-style abduction–action–prediction sequence over the latent state, which is a useful abstraction for both documentation and implementation.[file:1]

## Simulation environment

Simulation runs on a fidelity spectrum. The reason is economic: latent rollouts are cheap and abundant, while high-fidelity digital twins are expensive and should be reserved for validation, safety cases, and high-stakes decisions.[file:1]

### Fidelity levels

| Fidelity | Description | Typical purpose |
|---|---|---|
| Latent-only | Pure RSSM rollout without decoding | Fast policy improvement and exploration [file:1] |
| Periodic decode | Decode every N steps for plausibility checks | Human inspection and intermediate validation [file:1] |
| Classifier-guided | Latent rollout scored by learned outcome heads | Fast filtering and safety triage [file:1] |
| Symbolic simulation | State-machine or rule-driven execution | Logical and relational domains [file:1] |
| Digital twin | Full environment surrogate or clone | Final validation and safety-critical testing [file:1] |

The architecture should make fidelity selection a scheduler decision informed by uncertainty, compute budget, and deployment risk rather than a static configuration value.[file:1]

## Learning and policy integration

DreamEngine learns from both real and imagined data, but dream-derived policy changes must remain candidate updates until validated. This distinction should be explicit in the architecture because it is one of the main controls against hallucination drift and reward hacking.[file:1]

### Learning path

The learning path has two primary loops:

1. **World model refinement** from real experience and self-supervised objectives such as reconstruction, reward prediction, terminal prediction, and multi-step consistency.[file:1]
2. **Policy refinement** from imagined rollouts, using conservative value estimation and regularization against wake-policy drift.[file:1]

The paper recommends a staged deployment gate: shadow execution, canary rollout, gradual expansion, then full deployment if safety and performance criteria continue to pass. Rollback must be automatic when degradation, anomaly, or constraint violations are detected.[file:1]

### Stability controls

The architecture should include the following controls in the learning path:

- KL constraints that prevent excessive divergence from the wake policy.[file:1]
- Elastic Weight Consolidation to reduce catastrophic forgetting during dream-heavy updates.[file:1]
- Conservative or pessimistic value estimates when dream uncertainty is high.[file:1]
- Fidelity discriminators and consistency regularizers to reduce representational drift.[file:1]

## Orchestration model

DreamEngine maps well onto a supervisor/worker architecture. The scheduler, rollout controller, and validation policy should live in a logically centralized control plane, while dream generation, simulation, replay processing, and training run as distributed workers.[file:1]

For a Hermes–OpenClaw deployment model:

| Role | Responsibility |
|---|---|
| Hermes supervisor | Trigger evaluation, queueing, budget policy, fidelity selection, validation policy, rollout decisions |
| OpenClaw workers | Dream generation, retrieval jobs, simulation jobs, validators, memory maintenance, deployment tasks |
| Shared services | Redis, TimescaleDB, Qdrant, object storage, metrics, tracing |

This preserves clear authority boundaries. Workers remain disposable and scoped, while the supervisor maintains policy, state transitions, and auditability.

## Multi-agent extension

The paper describes four dream-sharing models, and the architecture should preserve them as separately configurable modes rather than one monolithic collective system.[file:1]

### Sharing models

| Model | Description | When to use |
|---|---|---|
| Private dreams | No sharing across agents | Sensitive, competitive, or regulated agents [file:1] |
| Shared dream pools | Store and retrieve collective dream artifacts | Teams learning from similar environments [file:1] |
| Hierarchical dreams | Team or organization-level strategic simulation | Coordinated fleets and shared strategy [file:1] |
| Federated dreaming | Privacy-preserving aggregate learning | Cross-org or privacy-sensitive collaboration [file:1] |

### MCP integration

MCP is used in the paper as the interoperability layer for dream exchange. Architecturally, dream artifacts should be exposed as typed resources, generation and validation should be exposed as tools, and shared prompt templates should standardize common dream requests such as adversarial scenario generation or counterfactual analysis.[file:1]

A dream artifact schema should minimally include:

- Dream identifier, agent identifier, timestamp, and dream type.[file:1]
- World model version and fidelity level.[file:1]
- Latent trajectory payload with actions, predicted rewards, values, and uncertainty.[file:1]
- Optional decoded observations.[file:1]
- Validation status and quality metrics.[file:1]
- Provenance linking source episodes, generator version, and seed state.[file:1]

## Security and alignment architecture

Security is not a wrapper around the dream system; it is part of the dream system itself. The document makes clear that dreaming introduces unique epistemic and operational risks because synthetic experience can recursively shape belief, planning, and policy.[file:1]

### Primary risks

| Risk class | Architectural impact |
|---|---|
| Confirmation bias loops | Dreams may reinforce bad priors unless diversity and failure replay are explicit [file:1] |
| Reality drift | World model error compounds across dream-policy cycles unless grounded by validation [file:1] |
| Dream poisoning | Shared pools require provenance, trust, and Byzantine-tolerant aggregation [file:1] |
| Latent adversarial attacks | Validation must include plausibility and anomaly checks [file:1] |
| Reward hacking | Dream-only gains must never bypass real validation [file:1] |
| Alignment drift | Values and constraints must be encoded into generation and evaluation layers [file:1] |

### Control requirements

The architecture should enforce:

- Provenance tags on every memory and dream artifact, including whether it is real, dreamed, validated, or rejected.[file:1]
- Multi-layer validation, from latent plausibility through symbolic consistency and, where needed, digital twin or empirical validation.[file:1]
- Trust scoring for dreams and dream sources, especially in shared or federated settings.[file:1]
- Human-in-the-loop checkpoints for high-impact dream classes and rollout decisions.[file:1]
- Namespace and network isolation so dreamed actions cannot affect production services.[file:1]

## Observability

DreamEngine should be instrumented as a distributed system, not a training script. At minimum, every dream request, rollout, validation decision, and policy promotion event should emit structured logs, metrics, and trace spans.[file:1]

Core observability domains include:

- Dream throughput, latency, and GPU time.[file:1]
- World model prediction error and drift metrics.[file:1]
- Dream utility metrics such as post-dream success improvement and sample-efficiency gain.[file:1]
- Validation failure categories and rollback events.[file:1]
- Scheduler decisions, queue state, and trigger frequency by class.[file:1]

TimescaleDB continuous aggregates and Redis stream consumers are a natural fit for near-real-time operations dashboards and anomaly detection.[file:1]

## Deployment topology

A production deployment should isolate wake, dream, validation, and shared services into separate Kubernetes namespaces. This mirrors the paper’s operational model and gives the platform clean boundaries for quotas, network policy, observability, and incident response.[file:1]

Suggested namespaces:

- `dreamengine-wake` for online inference and environment interaction.[file:1]
- `dreamengine-dream` for isolated simulation and training jobs.[file:1]
- `dreamengine-validate` for shadow and canary evaluation paths.[file:1]
- `dreamengine-shared` for memory, metrics, config, and coordination services.[file:1]

Ray should provide the distributed compute fabric, Kubernetes should provide placement and isolation, and the memory stores should scale independently based on their own access patterns.[file:1]

## Recommended initial implementation boundary

The architecture is broad, but the first production boundary should be narrow:

1. Single-agent only.[file:1]
2. RSSM world model only.[file:1]
3. Latent-only and periodic-decode simulation only.[file:1]
4. Redis + TimescaleDB + Qdrant only.[file:1]
5. No automatic policy promotion; shadow validation only.[file:1]

This subset is enough to prove the DreamEngine model without prematurely taking on hybrid generation, federated sharing, or digital-twin complexity. The rest of the architecture should remain visible in the document as extension points, but not as day-one implementation commitments.[file:1]
