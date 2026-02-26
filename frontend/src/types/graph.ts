export interface GraphNode {
  id: string;
  node_type: string;
  params: Record<string, unknown>;
  disabled: boolean;
  position: { x: number; y: number };
}

export interface GraphEdge {
  id: string;
  source_node: string;
  source_output: number;
  target_node: string;
  target_input: string;
  order: number;
}

export interface GraphSchema {
  nodes: GraphNode[];
  edges: GraphEdge[];
  name: string;
  description: string;
}

export interface ExecuteResponse {
  execution_id: string;
  status: string;
  results: Record<string, unknown[]>;
  errors: string[];
}

export interface TrainingProgress {
  type: string;
  epoch: number;
  total_epochs: number;
  train_loss: number;
  val_loss: number | null;
}
