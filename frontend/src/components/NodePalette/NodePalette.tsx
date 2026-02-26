import { useMemo } from 'react';
import type { NodeDefinition } from '../../types/nodes';
import { useGraphStore } from '../../store/graphStore';

interface Props {
  definitions: Record<string, NodeDefinition>;
}

let clickOffset = 0;

export function NodePalette({ definitions }: Props) {
  const addNode = useGraphStore((s) => s.addNode);
  const layerNodes = useMemo(() => {
    const items: { type: string; def: NodeDefinition }[] = [];
    for (const [type, def] of Object.entries(definitions)) {
      if (def.category === 'Layers') {
        items.push({ type, def });
      }
    }
    return items;
  }, [definitions]);

  const onDragStart = (event: React.DragEvent, nodeType: string) => {
    event.dataTransfer.setData('application/visdl-node', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div style={{
      width: 160,
      background: '#12121a',
      borderRight: '1px solid #2a2a3e',
      overflow: 'auto',
      padding: '12px 0',
      fontFamily: 'system-ui, sans-serif',
    }}>
      <div style={{
        padding: '0 12px 8px',
        color: '#6366f1',
        fontSize: 10,
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: 1,
      }}>
        Layers
      </div>
      {layerNodes.map(({ type, def }) => (
        <div
          key={type}
          draggable
          onDragStart={(e) => onDragStart(e, type)}
          onClick={() => {
            addNode(type, { x: 300 + clickOffset, y: 200 + clickOffset });
            clickOffset = (clickOffset + 40) % 200;
          }}
          style={{
            padding: '6px 12px 6px 16px',
            color: '#c0c0d0',
            fontSize: 12,
            cursor: 'grab',
            borderLeft: '3px solid transparent',
            transition: 'all 0.15s',
          }}
          onMouseEnter={(e) => {
            (e.target as HTMLElement).style.background = '#1e1e2e';
            (e.target as HTMLElement).style.borderLeftColor = '#6366f1';
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
  );
}
