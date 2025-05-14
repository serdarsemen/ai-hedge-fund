import { AppNode } from "@/nodes/types";

// Map of sidebar item names to node creation functions
export interface NodeTypeDefinition {
  type: string;
  createNode: (position: { x: number, y: number }) => AppNode;
}

// Define node creation functions for each type
const nodeTypeDefinitions: Record<string, NodeTypeDefinition> = {
  "Text Input": {
    type: "text-input-node",
    createNode: (position: { x: number, y: number }): AppNode => ({
      id: `start`,
      type: "text-input-node",
      position,
      data: {
        name: "Input",
        description: "Start Node",
        status: "Idle",
      },
    }),
  },
  "Ben Graham": {
    type: "agent",
    createNode: (position: { x: number, y: number }): AppNode => ({
      id: `ben_graham`,
      type: "agent",
      position,
      data: {
        name: "Ben Graham",
        description: "The Father of Value Investing",
        status: "Idle",
      },
    }),
  },
  "Charlie Munger": {
    type: "agent",
    createNode: (position: { x: number, y: number }): AppNode => ({
      id: `charlie_munger`,
      type: "agent",
      position,
      data: {
        name: "Charlie Munger",
        description: "The Abominable No-Man",
        status: "Idle",
      },
    }),
  },
  "Warren Buffett": {
    type: "agent",
    createNode: (position: { x: number, y: number }): AppNode => ({
      id: `warren_buffett`,
      type: "agent", 
      position,
      data: {
        name: "Warren Buffett",
        description: "The Oracle of Omaha",
        status: "Idle",
      },
    }),
  },
};

export function getNodeTypeDefinition(componentName: string): NodeTypeDefinition | null {
  return nodeTypeDefinitions[componentName] || null;
} 