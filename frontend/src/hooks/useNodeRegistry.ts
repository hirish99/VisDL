import { useEffect, useRef } from 'react';
import { fetchNodes } from '../api/client';
import { useGraphStore } from '../store/graphStore';

const DEFAULT_GRAPH = {
  nodes: [
    { id: 'l1', node_type: 'Linear', params: { out_features: 32 }, disabled: false, position: { x: 0, y: 0 } },
    { id: 'relu1', node_type: 'ReLU', params: {}, disabled: false, position: { x: 230, y: 0 } },
    { id: 'l2', node_type: 'Linear', params: { out_features: 16 }, disabled: false, position: { x: 420, y: 0 } },
    { id: 'relu2', node_type: 'ReLU', params: {}, disabled: false, position: { x: 650, y: 0 } },
    { id: 'l3', node_type: 'Linear', params: { out_features: 1 }, disabled: false, position: { x: 840, y: 0 } },
  ],
  edges: [
    { id: 'e1', source_node: 'l1', source_output: 0, target_node: 'relu1', target_input: 'prev_specs' },
    { id: 'e2', source_node: 'relu1', source_output: 0, target_node: 'l2', target_input: 'prev_specs' },
    { id: 'e3', source_node: 'l2', source_output: 0, target_node: 'relu2', target_input: 'prev_specs' },
    { id: 'e4', source_node: 'relu2', source_output: 0, target_node: 'l3', target_input: 'prev_specs' },
  ],
};

export function useNodeRegistry() {
  const setNodeDefinitions = useGraphStore((s) => s.setNodeDefinitions);
  const loadFromSchema = useGraphStore((s) => s.loadFromSchema);
  const defs = useGraphStore((s) => s.nodeDefinitions);
  const loaded = useRef(false);

  useEffect(() => {
    fetchNodes().then((nodeDefs) => {
      setNodeDefinitions(nodeDefs);
      if (!loaded.current) {
        loaded.current = true;
        setTimeout(() => {
          useGraphStore.getState().loadFromSchema(DEFAULT_GRAPH);
        }, 0);
      }
    }).catch(console.error);
  }, [setNodeDefinitions, loadFromSchema]);

  return defs;
}
