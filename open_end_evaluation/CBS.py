from pathlib import Path
from string import Template
PROMPT = Template(r"""
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
}
""")

def CBS(topology, flows, route):
    topo_txt = Path(topology).read_text(encoding="utf-8-sig").strip()
    flows_txt = Path(flows).read_text(encoding="utf-8-sig").strip()
    route_txt = Path(route).read_text(encoding="utf-8-sig").strip()
    return PROMPT.substitute(topology=topo_txt, flows=flows_txt, route=route_txt)