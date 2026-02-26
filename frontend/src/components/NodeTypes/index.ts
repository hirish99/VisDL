import type { ComponentType } from 'react';
import { BaseNode } from './BaseNode';
import { DataNode } from './DataNode';
import { TrainingNode } from './TrainingNode';
import { MetricsNode } from './MetricsNode';

// Map node_type → specialized component (fallback to BaseNode)
const SPECIALIZED: Record<string, ComponentType<any>> = {
  CSVLoader: DataNode,
  DataSplitter: BaseNode,
  TrainingLoop: TrainingNode,
  MetricsCollector: MetricsNode,
};

export function buildNodeTypes(
  nodeTypes: string[],
): Record<string, ComponentType<any>> {
  const result: Record<string, ComponentType<any>> = {};
  for (const nt of nodeTypes) {
    result[nt] = SPECIALIZED[nt] || BaseNode;
  }
  return result;
}
