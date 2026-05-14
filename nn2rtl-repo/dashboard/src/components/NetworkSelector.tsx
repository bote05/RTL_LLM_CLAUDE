import { Network } from "lucide-react";
import type { NetworkId } from "../shared/networks";
import type { NetworkInfo } from "../shared/types";

/** Top-of-page network picker.
 *
 * Drives `getSnapshot(networkId)` and every job action's `networkId`. Today
 * there is one available network (ResNet-50); future networks will appear
 * here automatically once they are added to `dashboard/src/shared/networks.ts`. */
export function NetworkSelector({ networks, value, onChange }: {
  networks: NetworkInfo[];
  value: NetworkId;
  onChange: (id: NetworkId) => void;
}): JSX.Element {
  const current = networks.find((network) => network.id === value);
  return (
    <div className="network-selector" title="Pick which neural network this dashboard view is bound to">
      <Network size={16} />
      <span className="muted">Network</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value as NetworkId)}
        aria-label="Neural network selector"
      >
        {networks.map((network) => (
          <option key={network.id} value={network.id} disabled={!network.available}>
            {network.label}{network.available ? "" : " (coming soon)"}
          </option>
        ))}
      </select>
      {current && <span className="network-desc muted">{current.description}</span>}
    </div>
  );
}
