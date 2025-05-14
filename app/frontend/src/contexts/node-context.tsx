import { createContext, ReactNode, useCallback, useContext, useState } from 'react';

export type NodeStatus = 'IDLE' | 'IN_PROGRESS' | 'COMPLETE' | 'ERROR';

// Message history item
export interface MessageItem {
  timestamp: string;
  message: string;
  ticker: string | null;
}

// Enhanced node state structure
export interface NodeData {
  status: NodeStatus;
  ticker: string | null;
  message: string;
  lastUpdated: number;
  messages: MessageItem[]; // History of all messages for this agent
  timestamp?: string; // Optional timestamp for the current state
}

interface NodeContextType {
  nodeStates: Record<string, NodeData>;
  updateNode: (nodeId: string, data: Partial<NodeData>) => void;
  updateNodeStatus: (nodeId: string, status: NodeStatus) => void;
  updateNodes: (nodeIds: string[], status: NodeStatus) => void;
  resetAllNodes: () => void;
}

const NodeContext = createContext<NodeContextType | undefined>(undefined);

export function NodeProvider({ children }: { children: ReactNode }) {
  const [nodeStates, setNodeStates] = useState<Record<string, NodeData>>({});

  const updateNode = useCallback((nodeId: string, data: Partial<NodeData>) => {
    setNodeStates((prev) => {
      // Get the existing node or create a default one
      const existingNode = prev[nodeId] || {
        status: 'IDLE',
        ticker: null,
        message: '',
        messages: [],
        lastUpdated: Date.now()
      };
      
      // If there's a new message, add it to the history
      const newMessages = [...existingNode.messages];
      if (data.message && data.message !== existingNode.message) {
        newMessages.push({
          timestamp: data.timestamp || new Date().toISOString(),
          message: data.message,
          ticker: data.ticker || existingNode.ticker
        });
      }
      
      return {
        ...prev,
        [nodeId]: {
          ...existingNode,
          ...data,
          messages: newMessages,
          lastUpdated: Date.now()
        }
      };
    });
  }, []);

  // Keep the old API for compatibility
  const updateNodeStatus = useCallback((nodeId: string, status: NodeStatus) => {
    updateNode(nodeId, { status });
  }, [updateNode]);

  // Update multiple nodes at once (keep for compatibility)
  const updateNodes = useCallback((nodeIds: string[], status: NodeStatus) => {
    setNodeStates((prev) => {
      const newStates = { ...prev };
      nodeIds.forEach((id) => {
        const existingNode = newStates[id] || {
          ticker: null,
          message: '',
          messages: [],
          lastUpdated: Date.now()
        };
        
        newStates[id] = {
          ...existingNode,
          status,
          lastUpdated: Date.now()
        } as NodeData;
      });
      return newStates;
    });
  }, []);

  const resetAllNodes = useCallback(() => {
    setNodeStates({});
  }, []);

  return (
    <NodeContext.Provider
      value={{
        nodeStates,
        updateNode,
        updateNodeStatus,
        updateNodes,
        resetAllNodes,
      }}
    >
      {children}
    </NodeContext.Provider>
  );
}

export function useNodeStatus() {
  const context = useContext(NodeContext);
  
  if (context === undefined) {
    throw new Error('useNodeStatus must be used within a NodeProvider');
  }
  
  return context;
} 