import { useMemo } from 'react';
import type { NodeDefinition } from '../../types/nodes';
import { useGraphStore } from '../../store/graphStore';

interface Props {
  definitions: Record<string, NodeDefinition>;
}

let clickOffset = 0;

export function NodePalette({ definitions }: Props) {
  const addNode = useGraphStore((s) => s.addNode);
  const grouped = useMemo(() => {
    const groups: Record<string, { type: string; def: NodeDefinition }[]> = {};
    for (const [type, def] of Object.entries(definitions)) {
      if (!groups[def.category]) groups[def.category] = [];
      groups[def.category].push({ type, def });
    }
    return groups;
  }, [definitions]);

  const categoryOrder = ['Data', 'Layers', 'Model', 'Loss', 'Optimizer', 'Training', 'Metrics'];
  const sorted = categoryOrder.filter((c) => c in grouped);
  // Add any unlisted categories at the end
  for (const c of Object.keys(grouped)) {
    if (!sorted.includes(c)) sorted.push(c);
  }

  const categoryColors: Record<string, string> = {
    Layers: '#6366f1',
    Data: '#22c55e',
    Loss: '#ef4444',
    Optimizer: '#f59e0b',
    Model: '#8b5cf6',
    Training: '#3b82f6',
    Metrics: '#06b6d4',
  };

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/visdl-node', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div style={{
      width: 220,
      background: '#12121a',
      borderRight: '1px solid #2a2a3e',
      overflow: 'auto',
      padding: '12px 0',
      fontFamily: 'system-ui, sans-serif',
    }}>
      <div style={{
        padding: '0 12px 8px',
        color: '#808090',
        fontSize: 11,
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: 1,
      }}>
        Nodes
      </div>
      {sorted.map((category) => (
        <div key={category} style={{ marginBottom: 12 }}>
          <div style={{
            padding: '4px 12px',
            color: categoryColors[category] || '#888',
            fontSize: 10,
            fontWeight: 600,
            textTransform: 'uppercase',
          }}>
            {category}
          </div>
          {grouped[category].map(({ type, def }) => (
            <div
              key={type}
              draggable
              onDragStart={(e) => onDragStart(e, type)}
              onClick={() => {
                addNode(type, { x: 300 + clickOffset, y: 200 + clickOffset });
                clickOffset = (clickOffset + 40) % 200;
              }}
              style={{
                padding: '6px 12px 6px 20px',
                color: '#c0c0d0',
                fontSize: 12,
                cursor: 'grab',
                borderLeft: `3px solid transparent`,
                transition: 'all 0.15s',
              }}
              onMouseEnter={(e) => {
                (e.target as HTMLElement).style.background = '#1e1e2e';
                (e.target as HTMLElement).style.borderLeftColor = categoryColors[category] || '#888';
              }}
              onMouseLeave={(e) => {
                (e.target as HTMLElement).style.background = 'transparent';
                (e.target as HTMLElement).style.borderLeftColor = 'transparent';
              }}
              title={def.description}
            >
              {def.display_name}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
