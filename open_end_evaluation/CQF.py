from pathlib import Path
from string import Template
PROMPT = Template(r"""
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
}
""")

def CQF(topology, flows, route):
    topo_txt = Path(topology).read_text(encoding="utf-8-sig").strip()
    flows_txt = Path(flows).read_text(encoding="utf-8-sig").strip()
    route_txt = Path(route).read_text(encoding="utf-8-sig").strip()
    return PROMPT.substitute(topology=topo_txt, flows=flows_txt, route=route_txt)