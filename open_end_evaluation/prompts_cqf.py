"""CQF prompts."""

from pathlib import Path
from string import Template

CLOSED_BOOK = Template(r"""
You are an expert Time-Sensitive Networking (TSN) orchestrator. Your task is to calculate the worst case delay (WCD) for each TSN flow.
Inputs:
Topology:${topology}
Flows:${flows}
Shortest Route:${route}
Each flow in Flows file is described by:<flow number>, <source>, <destination>, <periodicity (in microseconds)>, <deadline (in microseconds)>, <payload (in Bytes)>
Each route for the flows in Shortest Route file is described by:<flow number>, <source>, <destination>, <hop count in the route>, <route path>

Constant:
Bandwidth link = 100 Mbps ; Propagation delay = 1µs; Switching delay = 1µs; Time synchronization error = 1µs; The switches of the network are cut-through switches; cycle duration is 50µs.

1. TSN Mechanism:
Only Cyclic Queuing and Forwarding (CQF, IEEE 802.1Qch) is allowed;
All flows are TT, PCP = 7, using queue 7 (odd) and 6 (even) only.

Task:
1.Map each egress port's queues and collect the set of flows traversing that port, using the given topology, flows, and route of the flow.
2.For the entire network, use the given cycle duration and compute the Hypercycle.
3.For each flow, set the offset or the start time of the flow from the sending node as 0.
4.Calculate the worst case delay (WCD) in microseconds (µs) for each flow.
5.Provide the confidence score between 0.0 and 1.0 from your answers. 1.0 means mathematically/procedurally provable from given info with zero ambiguity. 0.0 means zero confidence.

Important:
1.do not change the given topology, flows and route;
2.Priority levels range from 0 to 7, with 0 being the lowest and 7 the highest;
3.answer strictly in JSON;
4.tell all the parameter that you are using for CQF;
5.give the equation you are using to calculate the WCD of the flows;

The output JSON MUST strictly follow this schema:

{
  "flow_profile": {
    "F0": ["CQF", "TT", "PCP:7"],
    "F1": ["CQF", "TT", "PCP:7"]
  },

  "WCD_us": {
    "F0": 0,"F1": 0,...
  },

  "reason": {
    "F0": {
      "path": ["..."],
      "cycle_duration":["..."],
      "Hypercycle":["..."],
      "cqf_parameters": {...},
      ...
        },
        "all_parameters_used": {....}
      },
      "WCD_calculation": {....},
      "missing_inputs": []
    }
  },
  "confidence": 0.5
}""")

FORMULA = Template(r"""
You are an expert Time-Sensitive Networking (TSN) orchestrator. Your task is to calculate the worst case delay (WCD) for each TSN flow.
Inputs:
Topology:${topology}
Flows:${flows}
Shortest Route:${route}
Each flow in Flows file is described by:<flow number>, <source>, <destination>, <periodicity (in microseconds)>, <deadline (in microseconds)>, <payload (in Bytes)>
Each route for the flows in Shortest Route file is described by:<flow number>, <source>, <destination>, <hop count in the route>, <route path>

Constant:
Bandwidth link = 100 Mbps ; Propagation delay = 1µs; Switching delay = 1µs; Time synchronization error = 1µs; The switches of the network are cut-through switches; cycle duration is 50µs.

1. TSN Mechanism:
Only Cyclic Queuing and Forwarding (CQF, IEEE 802.1Qch) is allowed;
All flows are TT, PCP = 7, using queue 7 (odd) and 6 (even) only.

CQF worst case delay (WCD) equation (use these directly; do not derive alternative formulations):
Hypercycle = LCM(all flow periods)
WCD of flow number i is: WCD_f_i = offset_i + (n + 1) * cycle duration + n * switching delay + propagation delay + time synchronization error
offset_i is the offset (start time) of flow i at its source node.
n is the number of switches in the route of flow i (n = hop count - 1).
Propagation delay and time synchronization error are added once per flow.

Task:
1.Map each egress port's queues and collect the set of flows traversing that port, using the given topology, flows, and route of the flow.
2.For the entire network, use the given cycle duration and compute the Hypercycle.
3.For each flow, set the offset or the start time of the flow from the sending node as 0.
4.Calculate the worst case delay (WCD) in microseconds (µs) for each flow.
5.Provide the confidence score between 0.0 and 1.0 from your answers. 1.0 means mathematically/procedurally provable from given info with zero ambiguity. 0.0 means zero confidence.

Important:
1.do not change the given topology, flows, priority of the flows, and route;
2.Priority levels range from 0 to 7, with 0 being the lowest and 7 the highest;
3.answer strictly in JSON;
4.tell all the parameter that you are using for CQF;
5.give the equation you are using to calculate the WCD of the flows;

The output JSON MUST strictly follow this schema:

{
  "flow_profile": {
    "F0": ["CQF", "TT", "PCP:7"],
    "F1": ["CQF", "TT", "PCP:7"]
  },

  "WCD_us": {
    "F0": 0,"F1": 0,...
  },

  "reason": {
    "F0": {
      "path": ["..."],
      "cycle_duration":["..."],
      "Hypercycle":["..."],
      "cqf_parameters": {...},
      ...
        },
        "all_parameters_used": {....}
      },
      "WCD_calculation": {....},
      "missing_inputs": []
    }
  },
  "confidence": 0.5
}""")

VARIANT_A = Template(r"""You are a senior network engineer specializing in deterministic Ethernet and real-time communication. You have been asked to determine the worst-case delay (WCD) that each flow in a Time-Sensitive Network will experience.

Given information:
Network topology:${topology}
Flow definitions:${flows}
Routing paths:${route}
Flow format: <flow number>, <source>, <destination>, <periodicity (µs)>, <deadline (µs)>, <payload (Bytes)>
Route format: <flow number>, <source>, <destination>, <hop count>, <route path>

Fixed parameters:
Bandwidth = 100 Mbps; Propagation delay = 1µs; Switching delay = 1µs; Time synchronization error = 1µs; switches operate in cut-through mode; Cycle duration = 50µs.

Mechanism constraint:
The only TSN mechanism in effect is Cyclic Queuing and Forwarding (CQF, IEEE 802.1Qch). Every flow is time-triggered (TT) traffic with PCP = 7, using only queue 7 (odd queue) and queue 6 (even queue).

What you need to do:
a. For every egress port, identify which queues the flows use and determine the complete set of flows sharing that port, based on the topology, flow list, and routing paths provided.
b. Using the given cycle duration, compute the Hypercycle for the entire network.
c. Set the offset, i.e. the start time of each flow at its sending node, to 0.
d. Compute the worst-case delay (in µs) for every flow.
e. Assign a confidence score from 0.0 to 1.0 to your result — 1.0 means the answer follows deterministically and unambiguously from the given information; 0.0 means you have no confidence in it.

Constraints on your answer:
- The provided topology, flow list, and routes must be used exactly as given, without modification.
- Priority levels span 0 (lowest) through 7 (highest).
- Respond in JSON only.
- State every CQF parameter you relied on.
- State the specific equation you used to compute each flow's WCD.

Your response must conform exactly to this JSON schema:

{
  "flow_profile": {
    "F0": ["CQF", "TT", "PCP:7"],
    "F1": ["CQF", "TT", "PCP:7"]
  },

  "WCD_us": {
    "F0": 0,"F1": 0,...
  },

  "reason": {
    "F0": {
      "path": ["..."],
      "cycle_duration":["..."],
      "Hypercycle":["..."],
      "cqf_parameters": {...},
      ...
        },
        "all_parameters_used": {....}
      },
      "WCD_calculation": {....},
      "missing_inputs": []
    }
  },
  "confidence": 0.5
}""")

VARIANT_B = Template(r"""
You are an expert Time-Sensitive Networking (TSN) orchestrator. Your task is to calculate the worst case delay (WCD) for each TSN flow.
Inputs:
Topology:${topology}
Flows:${flows}
Shortest Path:${route}
Each flow in Flows file is described by:<flow number>, <source>, <destination>, <periodicity (in microseconds)>, <deadline (in microseconds)>, <payload (in Bytes)>
Each route for the flows in Shortest Route file is described by:<flow number>, <source>, <destination>, <hop count in the route>, <route path>

Constant:
Bandwidth = 100 Mbps; Propagation delay = 1µs; Switching delay = 1µs; Time synchronization error = 1µs; The switches of the network are cut-through switches; cycle duration is 50µs..

1. TSN Mechanism:
Only Cyclic Queuing and Forwarding (CQF, IEEE 802.1Qch) is allowed;
All flows are TT, PCP = 7, using queue 7 (odd) and 6 (even) only.

Task:
1.Map each egress port's queues and collect the set of flows traversing from that port, using the given topology, flows, and route of the flow.
2.For the entire network, use the given cycle duration and compute the Hypercycle.
3.For each flow, set the offset or the start time of the flow from the sending node as 0.
4.Calculate the worst case delay (WCD) in microseconds (µs) for each flow.
5.Provide the confidence score between 0.0 and 1.0 from your answers. 1.0 means mathematically/procedurally provable from given info with zero ambiguity. 0.0 means zero confidence.

Important:
1.do not change the given topology, flows and route;
2.Priority levels range from 0 to 7, with 0 being the lowest and 7 the highest;
3.answer STRICTLY in the plain-text format specified below. Do NOT use JSON, markdown tables, or code blocks;
4.tell all the parameter that you are using for CQF;
5.give the equation you are using to calculate the WCD of the flows;

Your response MUST follow this exact plain-text structure, repeated once per flow, with no additional commentary before or after:

FLOW: <flow id>
MECHANISM: CQF | TT | PCP:7
PATH: <hop-by-hop path>
CYCLE_DURATION: <value used>
HYPERCYCLE: <value used>
CQF_PARAMETERS: <key=value, key=value, ...>
ALL_PARAMETERS_USED: <key=value, key=value, ...>
WCD_CALCULATION: <equation and substituted values used to compute WCD>
WCD_US: <final numeric WCD in microseconds>
MISSING_INPUTS: <comma-separated list, or NONE>
CONFIDENCE: <float between 0.0 and 1.0>
---""")

PROMPTS = {"closed_book": CLOSED_BOOK, "formula": FORMULA, "variant_a": VARIANT_A, "variant_b": VARIANT_B}


def CQF(topology, flows, route, variant="closed_book"):
    topo_txt = Path(topology).read_text(encoding="utf-8-sig").strip()
    flows_txt = Path(flows).read_text(encoding="utf-8-sig").strip()
    route_txt = Path(route).read_text(encoding="utf-8-sig").strip()
    return PROMPTS[variant].substitute(topology=topo_txt, flows=flows_txt, route=route_txt)
