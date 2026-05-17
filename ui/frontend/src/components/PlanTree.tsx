import { useMemo } from 'react';
import type { OperatorNode } from '../types';

interface PlanTreeProps {
  operators: OperatorNode[];
  highlight?: 'dop' | 'thread_block';
}

interface TreeNode {
  operator: OperatorNode;
  children: TreeNode[];
}

function buildForest(operators: OperatorNode[]): TreeNode[] {
  const map = new Map<number, TreeNode>();
  operators.forEach((op) => {
    map.set(op.plan_id, { operator: op, children: [] });
  });
  const roots: TreeNode[] = [];
  operators.forEach((op) => {
    const node = map.get(op.plan_id)!;
    if (op.parent_child === -1 || op.parent_child === null || op.parent_child === undefined) {
      roots.push(node);
      return;
    }
    const parent = map.get(op.parent_child);
    if (!parent) {
      roots.push(node);
    } else {
      parent.children.push(node);
    }
  });
  return roots;
}

function dopClass(dop: number): string {
  if (dop <= 1) return 'dop-1';
  if (dop <= 8) return 'dop-low';
  if (dop <= 32) return 'dop-mid';
  if (dop <= 64) return 'dop-high';
  return 'dop-max';
}

function renderNode(node: TreeNode, depth: number, highlight: 'dop' | 'thread_block'): JSX.Element {
  const op = node.operator;
  const classes = ['plan-tree-node'];
  if (highlight === 'dop') {
    classes.push(dopClass(op.dop));
  }
  if (depth === 0) classes.push('root');
  const indent = depth * 24;
  return (
    <div key={op.plan_id} style={{ marginLeft: indent, marginBottom: 4 }}>
      <div className={classes.join(' ')} title={`plan_id=${op.plan_id}`}>
        <span className="op">{op.operator_type}</span>
        <span className="meta">
          plan {op.plan_id} · dop {op.dop} · width {op.width}
          {highlight === 'thread_block' && op.thread_block_id !== undefined && op.thread_block_id !== null
            ? ` · tb ${op.thread_block_id}`
            : ''}
        </span>
      </div>
      {node.children.length > 0 && (
        <div>{node.children.map((child) => renderNode(child, depth + 1, highlight))}</div>
      )}
    </div>
  );
}

export default function PlanTree({ operators, highlight = 'dop' }: PlanTreeProps) {
  const roots = useMemo(() => buildForest(operators), [operators]);
  if (operators.length === 0) {
    return <div className="empty">暂无算子信息</div>;
  }
  return (
    <div className="plan-tree">
      {roots.map((root) => (
        <div key={root.operator.plan_id}>{renderNode(root, 0, highlight)}</div>
      ))}
    </div>
  );
}
