import { MessageItem } from '@/contexts/node-context';
import type { BuiltInNode, Node } from '@xyflow/react';

export type NodeMessage = MessageItem;

export type AgentNode = Node<{ name: string, description: string, status: string }, 'agent'>;
export type TextInputNode = Node<{ name: string, description: string, status: string }, 'text-input-node'>;
export type AppNode = BuiltInNode | AgentNode | TextInputNode;
