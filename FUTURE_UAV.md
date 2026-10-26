# Future UAV concepts — not demonstrated capabilities

<!-- Copyright (c) 2026 Tuomas Pietilä. SPDX-License-Identifier: CC-BY-4.0. Original expression only; upstream material and facts excluded. -->

**Developer and project lead: Tuomas Pietilä ([@Juurikko](https://github.com/Juurikko)).**

The verified milestone is on-device **file-based classification of already-georeferenced LAZ**. A mapping drone performing classification during flight is a proposed next system, not an accepted integration, streaming benchmark or flight-to-map result.

| Proposed capability | Potential use | Required engineering and experiment |
|---|---|---|
| Streaming onboard classification | Process bounded spatial/temporal windows during a mission | Integrate a sensor stream, synchronization and georeferencing; define chunk boundaries and overlap ownership; compare streaming labels against the qualified file path. |
| Compact semantic maps | Ground/vegetation/building/object-layer previews | Define aggregation, resolution, coordinate frames and completeness; measure information loss and bytes per update against full raw transfer. |
| Classification-aware telemetry | Send useful semantic deltas rather than entire raw clouds first | Implement bandwidth budgets, prioritization, dropped-packet recovery, versioning and stale-map detection; test constrained and interrupted links. |
| Confidence-based coverage assessment | Distinguish uncertain observations from missing coverage | Calibrate confidence and out-of-distribution behavior; track coverage provenance and georeferencing error. Argmax labels alone do not supply reliable uncertainty. |
| Targeted in-flight revisits | Request another observation of an uncertain area before landing | Validate navigation interfaces, operator oversight, mission constraints and revisit-selection value. First test offline replay, then simulation and controlled integration. |
| Reduced raw-data transfer dependence | Produce an early actionable preview while retaining raw records for later analysis | Measure preview utility, latency, operator decisions and omitted-detail cost; raw retention and auditability remain necessary. |

## Evidence needed before stronger claims

1. **Correctness under streaming:** deterministic windowing, stable source IDs, boundary handling and measured agreement with a complete-file reference. Georeferencing and motion compensation must be specified; they are not supplied by this classifier.
2. **Bounded latency and memory:** worst-case service time, queue backlog, backpressure and deadline behavior under realistic arrival rates and representative scenes. Full-file points/s alone does not establish these bounds.
3. **Confidence and coverage:** calibrated scores, unknown/missing categories, spatial coverage accounting, failure cases and false-revisit rates. A semantic class is not a confidence estimate.
4. **Onboard qualification:** sustained power/thermal tests, enclosure and payload constraints, vibration, storage endurance, energy budgets and competing flight-computer workloads. MODE_50W is a profile label, not a measured payload power budget.
5. **Telemetry/operator evaluation:** link throughput, map compression, update consistency, operator readability and measured time-to-useful-information.
6. **Mission integration:** controlled trials demonstrating that a proposed revisit improves mapping value without violating the mission envelope. No autonomous-flight capability or airborne safety qualification is claimed here.

The implementation should retain the evidence discipline of the file pipeline: fixed contracts, source/model identities, explicit timing boundaries, independent output checks and a distinction between measurements and illustrations. None of the experiments above were run for this repository.
