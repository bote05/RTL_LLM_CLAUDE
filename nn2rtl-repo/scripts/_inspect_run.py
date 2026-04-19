import json, sys
for i, line in enumerate(open(sys.argv[1]), 1):
    e = json.loads(line)
    ts = (e.get("timestamp") or "")[11:19]
    evt = e.get("event", "?")
    mod = e.get("module_id", "")
    act = e.get("action", "")
    agent = e.get("agent", "")
    reason = e.get("reason", "")
    print(f"{i:2d}. {ts}  {evt:30s}  mod={mod}  act={act} agent={agent} reason={reason}")
