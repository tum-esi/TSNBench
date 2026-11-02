"""CBS prompts."""

from pathlib import Path
from string import Template

CLOSED_BOOK = Template(r"""
You are an expert Time-Sensitive Networking (TSN) orchestrator. Your task is to calculate the worst case delay (WCD) for each TSN flow.
Inputs:
Topology:${topology}
Flows:${flows}
Shortest Path:${route}
Each flow in Flows file is described by:<flow number>, <source>, <destination>, <periodicity (in microseconds)>, <deadline (in microseconds)>, <payload (in Bytes)>
Each route for the flows in Shortest Route file is described by:<flow number>, <source>, <destination>, <hop count in the route>, <route path>

Constant:
Bandwidth = 100 Mbps; Propagation delay = 1µs; Switching delay = 1µs; Time synchronization error = 1µs; The switches of the network are cut-through switches; IdleSlope = 75%.

1. TSN Mechanism:
Only Credit-Based Shaper (CBS, IEEE 802.1Qav) is allowed;
All flows are AVB Class A, PCP = 6, using queue 6 only.

Task:
1.Map each egress port's queues and collect the set of flows traversing from that port, using the given topology, flows, and route of the flow.
2.For each egress port, use the given IdleSlope and then compute the SendSlope.
3.For each flow, construct an arrival curve from its frame size and periodicity.
4.For each port, derive a lower-bounded CBS service curve.
5.Calculate the worst case delay (WCD) in microseconds (µs) for each flow using Network Calculus method.
6.Provide the confidence score between 0.0 and 1.0 from your answers. 1.0 means mathematically/procedurally provable from given info with zero ambiguity. 0.0 means zero confidence.

Important:
1.do not change the given topology, flows and route;
2.Priority levels range from 0 to 7, with 0 being the lowest and 7 the highest;
3.answer strictly in JSON;
4.tell all the parameter that you are using for CBS;
5.give the equation you are using to calculate the WCD of the flows;

The output JSON MUST strictly follow this schema:

{
  "flow_profile": {
    "F0": ["CBS", "AVB_A", "PCP:6"],
    "F1": ["CBS", "AVB_A", "PCP:6"]
  },

  "WCD_us": {
    "F0": 0,"F1": 0,...
  },

  "reason": {
    "F0": {
      "path": ["..."],
      "IdleSlope":["..."],
      "SendSlope":["..."],
      "cbs_parameters": {...},
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
Shortest Path:${route}
Each flow in Flows file is described by:<flow number>, <source>, <destination>, <periodicity (in microseconds)>, <deadline (in microseconds)>, <payload (in Bytes)>
Each route for the flows in Shortest Route file is described by:<flow number>, <source>, <destination>, <hop count in the route>, <route path>

Constant:
Bandwidth = 100 Mbps; Propagation delay = 1µs; Switching delay = 1µs; Time synchronization error = 1µs; The switches of the network are cut-through switches; IdleSlope = 75%.

1. TSN Mechanism:
Only Credit-Based Shaper (CBS, IEEE 802.1Qav) is allowed;
All flows are AVB Class A, PCP = 6, using queue 6 only.

Network Calculus Equations (use these directly; do not derive alternative formulations):

Arrival curve, for flow f with frame size l_f and period P_f at its source node h0:
  alpha_f^h0(t) = b_f^h0 + rho_f^h0 * t,   for t > 0, else 0
  where b_f^h0 = l_f,  rho_f^h0 = l_f / P_f

Arrival curve propagation to an intermediate node h (output arrival curve of f departing server h-):
  alpha_f^h(t) = alpha_f^(h-) (deconvolved with) delta_{D_f^(h-)}(t)
  where D_f^(h-) is the latency upper bound of flow f at server h-, and delta_D(t) is the pure-delay function.

Aggregate arrival curve for all flows of priority class M_i at port h:
  alpha_{M_i}^h(t) = sum over incoming ports h- and flows f in class M_i of [ alpha_{M_i}^{h-,h}(t) AND link-shaping-curve AND CBS-shaping-curve ]
  where AND denotes min(x, y).

Link shaping curve from h- to h:
  sigma_link^{h-,h}(t) = C*t + l_{M_i}^{h-,h,max}
  where C is the link rate and l_{M_i}^{h-,h,max} is the max frame size of class M_i flows from h- to h.

CBS shaping curve of class M_i from h- to h:
  sigma_{M_i}^{h-,h}(t) = idleSlope_{M_i} * t + (c_{M_i}^{h-,max} - c_{M_i}^{h-,min}) + l_{M_i}^{h-,h,max}

Credit upper bound for class M_i at port h:
  c_{M_i}^{h,max} = idleSlope_{M_i}^h * ( sum_{j<i} c_{M_j}^{h,min} - l_{>i}^{h,max} ) / ( sum_{j<i} idleSlope_{M_j}^h - C )
  where l_{>i}^{h,max} is the max frame size of any class with priority lower than M_i (or best-effort) at h, j<i means the AVB class j with priority higher than i.

Credit lower bound for class M_i at port h:
  c_{M_i}^{h,min} = sendSlope_{M_i}^h * l_{M_i}^{h,max} / C
where sendSlope_{M_i}^h=idleSlope_{M_i}^h-C.

CBS service curve (rate-latency form) for class M_i at port h:
  beta_{M_i}^h(t) = idleSlope_{M_i}^h * [ t - c_{M_i}^{h,max} / idleSlope_{M_i}^h ]^+_{uparrow}
where [f(t)]^+_{uparrow}=max_{0<=s<=t}\{f(s), 0\}.

Worst-case per-hop delay at port h (horizontal deviation, NC Total Flow Analysis):
  D_f^h = D_{M_i}^h = hDev(alpha_{M_i}^h, beta_{M_i}^h)
        = sup_{t>=0} { inf { tau >= 0 : alpha_{M_i}^h(t) <= beta_{M_i}^h(t + tau) } }

End-to-end WCD for flow f:
  WCD_f = sum of D_f^h over every port h along f's route (as given in Path input),
          plus propagation delay, and switching delay per hop as specified in Constants.

Note: since idleSlope = sendSlope's complement of link rate implicitly (idleSlope + sendSlope maps to full reserved bandwidth), derive sendSlope_{M_i}^h from the given IdleSlope = 75% and Bandwidth = 100 Mbps as needed.

Task:
1.Map each egress port's queues and collect the set of flows traversing from that port, using the given topology, flows, and route of the flow.
2.For each egress port, use the given IdleSlope and then compute the SendSlope.
3.For each flow, construct an arrival curve from its frame size and periodicity.
4.For each port, derive a lower-bounded CBS service curve.
5.Calculate the worst case delay (WCD) in microseconds (µs) for each flow using Network Calculus method.
6.Provide the confidence score between 0.0 and 1.0 from your answers. 1.0 means mathematically/procedurally provable from given info with zero ambiguity. 0.0 means zero confidence.

Important:
1.do not change the given topology, flows and route;
2.Priority levels range from 0 to 7, with 0 being the lowest and 7 the highest;
3.answer strictly in JSON;
4.tell all the parameter that you are using for CBS;
5.give the equation you are using to calculate the WCD of the flows;

The output JSON MUST strictly follow this schema:

{
  "flow_profile": {
    "F0": ["CBS", "AVB_A", "PCP:6"],
    "F1": ["CBS", "AVB_A", "PCP:6"]
  },

  "WCD_us": {
    "F0": 0,"F1": 0,...
  },

  "reason": {
    "F0": {
      "path": ["..."],
      "IdleSlope":["..."],
      "SendSlope":["..."],
      "cbs_parameters": {...},
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
Bandwidth = 100 Mbps; Propagation delay = 1µs; Switching delay = 1µs; Time synchronization error = 1µs; switches operate in cut-through mode; IdleSlope = 75%.

Mechanism constraint:
The only shaping mechanism in effect is Credit-Based Shaper (CBS, IEEE 802.1Qav). Every flow belongs to AVB Class A, with PCP = 6, mapped to queue 6 exclusively.

What you need to do:
a. For every egress port, identify which queue each flow uses and determine the complete set of flows sharing that port, based on the topology, flow list, and routing paths provided.
b. Determine the SendSlope at each egress port from the given IdleSlope.
c. Build an arrival curve for each flow based on its frame size and periodicity.
d. Derive a lower-bounded CBS service curve for each port.
e. Using Network Calculus, compute the worst-case delay (in µs) for every flow.
f. Assign a confidence score from 0.0 to 1.0 to your result — 1.0 means the answer follows deterministically and unambiguously from the given information; 0.0 means you have no confidence in it.

Constraints on your answer:
- The provided topology, flow list, and routes must be used exactly as given, without modification.
- Priority levels span 0 (lowest) through 7 (highest).
- Respond in JSON only.
- State every CBS parameter you relied on.
- State the specific equation you used to compute each flow's WCD.

Your response must conform exactly to this JSON schema:

{
  "flow_profile": {
    "F0": ["CBS", "AVB_A", "PCP:6"],
    "F1": ["CBS", "AVB_A", "PCP:6"]
  },

  "WCD_us": {
    "F0": 0,"F1": 0,...
  },

  "reason": {
    "F0": {
      "path": ["..."],
      "IdleSlope":["..."],
      "SendSlope":["..."],
      "cbs_parameters": {...},
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
Bandwidth = 100 Mbps; Propagation delay = 1µs; Switching delay = 1µs; Time synchronization error = 1µs; The switches of the network are cut-through switches; IdleSlope = 75%.

1. TSN Mechanism:
Only Credit-Based Shaper (CBS, IEEE 802.1Qav) is allowed;
All flows are AVB Class A, PCP = 6, using queue 6 only.

Task:
1.Map each egress port's queues and collect the set of flows traversing from that port, using the given topology, flows, and route of the flow.
2.For each egress port, use the given IdleSlope and then compute the SendSlope.
3.For each flow, construct an arrival curve from its frame size and periodicity.
4.For each port, derive a lower-bounded CBS service curve.
5.Calculate the worst case delay (WCD) in microseconds (µs) for each flow using Network Calculus method.
6.Provide the confidence score between 0.0 and 1.0 from your answers. 1.0 means mathematically/procedurally provable from given info with zero ambiguity. 0.0 means zero confidence.

Important:
1.do not change the given topology, flows and route;
2.Priority levels range from 0 to 7, with 0 being the lowest and 7 the highest;
3.answer STRICTLY in the plain-text format specified below. Do NOT use JSON, markdown tables, or code blocks;
4.tell all the parameter that you are using for CBS;
5.give the equation you are using to calculate the WCD of the flows;

Your response MUST follow this exact plain-text structure, repeated once per flow, with no additional commentary before or after:

FLOW: <flow id>
MECHANISM: CBS | AVB_A | PCP:6
PATH: <hop-by-hop path>
IDLE_SLOPE: <value used>
SEND_SLOPE: <value used>
CBS_PARAMETERS: <key=value, key=value, ...>
ALL_PARAMETERS_USED: <key=value, key=value, ...>
WCD_CALCULATION: <equation and substituted values used to compute WCD>
WCD_US: <final numeric WCD in microseconds>
MISSING_INPUTS: <comma-separated list, or NONE>
CONFIDENCE: <float between 0.0 and 1.0>
---""")

PROMPTS = {"closed_book": CLOSED_BOOK, "formula": FORMULA, "variant_a": VARIANT_A, "variant_b": VARIANT_B}


def CBS(topology, flows, route, variant="closed_book"):
    topo_txt = Path(topology).read_text(encoding="utf-8-sig").strip()
    flows_txt = Path(flows).read_text(encoding="utf-8-sig").strip()
    route_txt = Path(route).read_text(encoding="utf-8-sig").strip()
    return PROMPTS[variant].substitute(topology=topo_txt, flows=flows_txt, route=route_txt)
