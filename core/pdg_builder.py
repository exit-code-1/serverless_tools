"""
Pipeline Dependency Graph (PDG) Builder.
Converts Stage DAG (ThreadBlocks) to PDG (Pipelines) based on pipeline breakers.
"""

from typing import Dict, List, Optional, Tuple
from core.thread_block import ThreadBlock
from core.plan_node import PlanNode
from core.pipeline_pair import Segment


class Pipeline:
    """
    Represents a single pipeline in the PDG.
    A pipeline is identified as Stage.Pipeline (e.g., S1.P1).
    """
    
    def __init__(self, stage_id: int, pipeline_id: int, nodes: List[PlanNode], latency: float = 0.0):
        """
        Initialize a Pipeline.
        
        Args:
            stage_id: Stage identifier (thread_id from ThreadBlock)
            pipeline_id: Pipeline identifier within the stage
            nodes: List of PlanNode objects in this pipeline
            latency: Pipeline latency t(p)
        """
        self.stage_id = stage_id
        self.pipeline_id = pipeline_id
        self.nodes = nodes
        self.latency = latency
        self.upstream_pipelines = []  # List of upstream Pipeline objects
        self.downstream_pipelines = []  # List of downstream Pipeline objects
    
    def __repr__(self):
        return f"S{self.stage_id}.P{self.pipeline_id}"
    
    def add_upstream(self, upstream_pipeline):
        """Add an upstream pipeline dependency."""
        if upstream_pipeline not in self.upstream_pipelines:
            self.upstream_pipelines.append(upstream_pipeline)
        if self not in upstream_pipeline.downstream_pipelines:
            upstream_pipeline.downstream_pipelines.append(self)


def find_pipeline_breakers(nodes: List[PlanNode]) -> List[PlanNode]:
    """
    Find all pipeline breakers in a stage.
    Pipeline breakers are blocking operators (materialized nodes) that can be split
    into build and probe phases (e.g., hash join, aggregate).
    
    Args:
        nodes: List of nodes in the stage
        
    Returns:
        List of pipeline breaker nodes, sorted by their position in the stage
    """
    breakers = []
    for node in nodes:
        if node.materialized:
            breakers.append(node)
    # Sort by plan_id or position to maintain order
    breakers.sort(key=lambda n: n.plan_id)
    return breakers


def build_segments_from_bottom_up(all_nodes: List[PlanNode], thread_blocks: Dict[int, ThreadBlock]) -> Segment:
    """
    Build segments from bottom to top.
    Start from leaf nodes, traverse upward until hitting a pipeline breaker.
    A segment is either:
    - Inner-stage: nodes within a single stage (not crossing stage boundary)
    - Cross-stage: nodes spanning multiple stages (crossing streaming operator)
    
    Args:
        all_nodes: All nodes in the query plan
        thread_blocks: Dict mapping thread_id to ThreadBlock
        
    Returns:
        Top-level Segment representing the entire query
    """
    # #region agent log
    import json
    log_path = '/home/zhy/opengauss/tools/new_serverless_predictor/.cursor/debug.log'
    with open(log_path, 'a') as f:
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"pdg_builder.py:82","message":"build_segments_from_bottom_up entry","data":{"all_nodes_count":len(all_nodes),"thread_blocks_count":len(thread_blocks)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Find leaf nodes (nodes with no children, e.g., CStore Scan)
    # Start from leaf nodes only, then traverse layer by layer
    leaf_nodes = [node for node in all_nodes if not node.child_plans]
    
    # #region agent log
    with open(log_path, 'a') as f:
        leaf_info = [{"plan_id":n.plan_id,"operator_type":n.operator_type,"materialized":n.materialized} for n in leaf_nodes]
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"E","location":"pdg_builder.py:84","message":"leaf nodes found","data":{"leaf_count":len(leaf_nodes),"leaf_nodes":leaf_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # Log tree structure
        tree_structure = []
        for node in all_nodes:
            parent_id = node.parent_node.plan_id if node.parent_node else None
            child_ids = [c.plan_id for c in node.child_plans] if node.child_plans else []
            tree_structure.append({
                "plan_id": node.plan_id,
                "operator_type": node.operator_type,
                "parent_id": parent_id,
                "child_ids": child_ids,
                "materialized": node.materialized
            })
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"F","location":"pdg_builder.py:97","message":"tree structure","data":{"tree_structure":tree_structure},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Map node to its thread_id (stage_id)
    node_to_stage = {}
    for thread_id, tb in thread_blocks.items():
        for node in tb.nodes:
            node_to_stage[node] = thread_id
    
    segments = []  # List of all segments
    processed_nodes = set()  # Track which nodes have been processed
    
    def is_streaming_operator(node: PlanNode) -> bool:
        """Check if node is a streaming operator (stage boundary)"""
        return 'streaming' in node.operator_type.lower()
    
    def traverse_upward_to_breaker(start_node: PlanNode) -> Tuple[List[PlanNode], Optional[PlanNode], bool]:
        """
        Traverse upward from start_node until hitting a pipeline breaker.
        The segment includes nodes up to and including the breaker.
        
        Returns:
            Tuple of:
            - nodes_in_segment: List of nodes in this segment (up to and including breaker)
            - breaker: The breaker node (if found), None otherwise
            - is_cross_stage: True if segment spans multiple stages
        """
        nodes_in_segment = [start_node]
        current = start_node
        start_stage_id = node_to_stage.get(start_node, 0)
        is_cross_stage = False
        
        # Traverse upward
        while current.parent_node:
            parent = current.parent_node
            
            # Check if parent is a pipeline breaker - include it and stop
            if parent.materialized:
                # Found a breaker, include it in the segment
                nodes_in_segment.append(parent)
                # Check if we crossed a stage boundary
                parent_stage_id = node_to_stage.get(parent, 0)
                if parent_stage_id != start_stage_id or is_streaming_operator(current):
                    is_cross_stage = True
                return nodes_in_segment, parent, is_cross_stage
            
            # Check if we crossed a stage boundary (streaming operator)
            parent_stage_id = node_to_stage.get(parent, 0)
            if parent_stage_id != start_stage_id or is_streaming_operator(current):
                is_cross_stage = True
            
            nodes_in_segment.append(parent)
            current = parent
        
        # Reached root without finding a breaker
        return nodes_in_segment, None, is_cross_stage
    
    # Layer-by-layer traversal: start from unprocessed leaf nodes, then process breaker's parents level by level
    from collections import deque
    queue = deque(leaf_nodes)  # Initialize queue with leaf nodes only
    
    while queue:
        current_node = queue.popleft()
        
        if current_node in processed_nodes:
            continue
        
        # #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"pdg_builder.py:135","message":"processing node","data":{"node_plan_id":current_node.plan_id,"node_operator":current_node.operator_type},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        # Traverse upward to find segment (up to and including breaker)
        segment_nodes, breaker, is_cross_stage = traverse_upward_to_breaker(current_node)
        
        if not segment_nodes:
            continue
        
        # Mark nodes as processed
        for node in segment_nodes:
            processed_nodes.add(node)
        
        # #region agent log
        with open(log_path, 'a') as f:
            segment_node_info = [{"plan_id":n.plan_id,"operator_type":n.operator_type,"materialized":n.materialized} for n in segment_nodes]
            breaker_info = {"plan_id":breaker.plan_id,"operator_type":breaker.operator_type,"materialized":breaker.materialized} if breaker else None
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"pdg_builder.py:151","message":"segment created","data":{"segment_nodes":segment_node_info,"is_cross_stage":is_cross_stage,"has_breaker":breaker is not None,"breaker":breaker_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        # Create segment (now includes breaker if found)
        stage_ids_in_segment = set()
        for node in segment_nodes:
            stage_ids_in_segment.add(node_to_stage.get(node, 0))
        
        # Calculate pipeline latency for each stage in the segment
        # pipeline_latencies should contain one latency per stage (per pipeline)
        # For cross-stage segments, different stages' pipelines run in parallel (max relationship)
        # For inner-stage segments, there's only one pipeline latency
        pipeline_latencies = []
        dop_info = {}
        for stage_id in stage_ids_in_segment:
            tb = thread_blocks.get(stage_id)
            if tb and tb.nodes:
                stage_nodes = [n for n in segment_nodes if node_to_stage.get(n, 0) == stage_id]
                if stage_nodes:
                    # Calculate pipeline latency for this stage: sum of nodes in this stage
                    stage_pipeline_latency = sum(n.execution_time for n in stage_nodes)
                    pipeline_latencies.append(stage_pipeline_latency)
                    dop_info[stage_id] = {
                        'dop': stage_nodes[0].dop if stage_nodes else 1,
                        'nodes': stage_nodes
                    }
        
        # #region agent log
        with open(log_path, 'a') as f:
            # Log node order and stage assignment
            node_order = [(n.plan_id, n.operator_type, node_to_stage.get(n, 0), n.execution_time) for n in segment_nodes]
            stage_latencies = {stage_id: sum(n.execution_time for n in segment_nodes if node_to_stage.get(n, 0) == stage_id) for stage_id in stage_ids_in_segment}
            # Log which nodes belong to which stage pipeline
            stage_nodes_detail = {}
            for stage_id in stage_ids_in_segment:
                stage_nodes = [n for n in segment_nodes if node_to_stage.get(n, 0) == stage_id]
                stage_nodes_detail[stage_id] = [(n.plan_id, n.operator_type, n.execution_time) for n in stage_nodes]
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"H","location":"pdg_builder.py:199","message":"segment latency calculation","data":{"pipeline_latencies":pipeline_latencies,"node_count":len(segment_nodes),"node_order":node_order,"stage_ids":list(stage_ids_in_segment),"stage_latencies":stage_latencies,"stage_nodes_detail":stage_nodes_detail,"is_cross_stage":is_cross_stage},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        segment = Segment(
            pipeline_latencies=pipeline_latencies,
            is_inner_stage=len(stage_ids_in_segment) == 1 and not is_cross_stage,
            nodes=segment_nodes,
            dop_info=dop_info,
            pipelines=[]
        )
        segments.append(segment)
        
        # After processing a segment, if we found a breaker, add its parent as next starting point
        # Only if the parent hasn't been visited yet (not in processed_nodes)
        # This ensures we continue from each unvisited breaker's parent node (e.g., probe after build)
        # Note: If the parent was already included in another path, it won't be added as a starting point
        if breaker and breaker.parent_node:
            parent = breaker.parent_node
            # Only add if parent hasn't been processed and isn't already in queue
            if parent not in processed_nodes and parent not in queue:
                # #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"I","location":"pdg_builder.py:252","message":"adding breaker parent to queue","data":{"breaker":(breaker.plan_id,breaker.operator_type),"parent":(parent.plan_id,parent.operator_type)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                queue.append(parent)
    
    # Process any remaining unprocessed nodes (shouldn't happen in a tree, but handle edge cases)
    unprocessed = [node for node in all_nodes if node not in processed_nodes]
    # #region agent log
    if unprocessed:
        with open(log_path, 'a') as f:
            unprocessed_info = [{"plan_id":n.plan_id,"operator_type":n.operator_type,"materialized":n.materialized} for n in unprocessed]
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"C","location":"pdg_builder.py:248","message":"unprocessed nodes found","data":{"unprocessed_count":len(unprocessed),"unprocessed_nodes":unprocessed_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    if unprocessed:
        # These might be nodes that are not reachable from leaves (shouldn't happen in valid query plans)
        # For now, create segments for them
        for node in unprocessed:
            segment_nodes, breaker, is_cross_stage = traverse_upward_to_breaker(node)
            for n in segment_nodes:
                processed_nodes.add(n)
            
            segment_latency = sum(n.execution_time for n in segment_nodes)
            stage_ids_in_segment = set()
            for n in segment_nodes:
                stage_id = node_to_stage.get(n, 0)
                stage_ids_in_segment.add(stage_id)
            
            is_inner = len(stage_ids_in_segment) == 1 and not is_cross_stage
            
            dop_info = {}
            for stage_id in stage_ids_in_segment:
                tb = thread_blocks.get(stage_id)
                if tb and tb.nodes:
                    stage_nodes = [n for n in segment_nodes if node_to_stage.get(n, 0) == stage_id]
                    if stage_nodes:
                        dop_info[stage_id] = {
                            'dop': stage_nodes[0].dop if stage_nodes else 1,
                            'nodes': stage_nodes
                        }
            
            segment = Segment(
                pipeline_latencies=[segment_latency],
                is_inner_stage=is_inner,
                nodes=segment_nodes,
                dop_info=dop_info,
                pipelines=[]
            )
            segments.append(segment)
    
    # #region agent log
    with open(log_path, 'a') as f:
        segments_info = [{"index":i,"nodes_count":len(s.nodes),"nodes":[(n.plan_id,n.operator_type) for n in s.nodes],"pipeline_latencies":s.pipeline_latencies} for i,s in enumerate(segments)]
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"A","location":"pdg_builder.py:260","message":"segments before dependency building","data":{"segments_count":len(segments),"segments":segments_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # Build upstream dependencies between segments
    # A segment depends on segments that produce data it consumes
    # For each segment, check if its top_node (breaker or last node) has a parent in another segment
    for seg in segments:
        if seg.nodes:
            # The last node in this segment (topmost, usually the breaker) - this is where data exits
            top_node = seg.nodes[-1]
            
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"pdg_builder.py:311","message":"checking segment for dependencies","data":{"seg_index":segments.index(seg),"top_node":(top_node.plan_id,top_node.operator_type),"has_parent":top_node.parent_node is not None,"parent":(top_node.parent_node.plan_id,top_node.parent_node.operator_type) if top_node.parent_node else None},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            
            # Check if top_node has a parent that consumes data from this segment
            # The parent node should be in a downstream segment
            # But we're building dependencies from the consumer's perspective:
            # If segment A's top_node has a parent in segment B, then segment B depends on segment A
            if top_node.parent_node:
                parent = top_node.parent_node
                # #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"pdg_builder.py:267","message":"building dependency","data":{"seg_index":segments.index(seg),"top_node":(top_node.plan_id,top_node.operator_type),"parent":(parent.plan_id,parent.operator_type)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
                # Find segment that contains the parent node (downstream segment)
                # First check if parent is in current segment - if so, skip (no dependency needed)
                if parent in seg.nodes:
                    # #region agent log
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"pdg_builder.py:267","message":"parent in same segment, skipping","data":{"seg_index":segments.index(seg),"parent":(parent.plan_id,parent.operator_type)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                    # #endregion
                    continue
                
                found_downstream = False
                for other_seg in segments:
                    if other_seg != seg and other_seg.nodes:
                        # Check if parent is in other_seg's nodes
                        if parent in other_seg.nodes:
                            # other_seg depends on seg (other_seg consumes data from seg)
                            other_seg.add_upstream_segment(seg)
                            found_downstream = True
                            # #region agent log
                            with open(log_path, 'a') as f:
                                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"pdg_builder.py:274","message":"dependency found","data":{"upstream_seg_index":segments.index(seg),"downstream_seg_index":segments.index(other_seg)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                            # #endregion
                            break
                
                
                # #region agent log
                if not found_downstream:
                    with open(log_path, 'a') as f:
                        # Check all segments to see where parent might be
                        all_seg_nodes = []
                        for other_seg in segments:
                            if other_seg.nodes:
                                all_seg_nodes.extend([(n.plan_id, n.operator_type) for n in other_seg.nodes])
                        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"B","location":"pdg_builder.py:276","message":"dependency NOT found","data":{"seg_index":segments.index(seg),"top_node":(top_node.plan_id,top_node.operator_type),"parent":(parent.plan_id,parent.operator_type),"parent_in_any_segment":parent in [n for s in segments for n in s.nodes],"all_segment_nodes":all_seg_nodes},"timestamp":int(__import__('time').time()*1000)}) + '\n')
                # #endregion
    
    # Find root segments: segments whose top_node has no parent_node (or parent not in any other segment)
    # Root seg = segment whose top_node has no parent_node
    root_segments = []
    for seg in segments:
        if seg.nodes:
            top_node = seg.nodes[-1]
            # If top_node has no parent, this is a root segment
            if not top_node.parent_node:
                root_segments.append(seg)
            else:
                # Check if top_node's parent is in any other segment
                parent = top_node.parent_node
                parent_in_other_segment = False
                for other_seg in segments:
                    if other_seg != seg and other_seg.nodes and parent in other_seg.nodes:
                        parent_in_other_segment = True
                        break
                # If parent is not in any other segment, this is also a root segment
                if not parent_in_other_segment:
                    root_segments.append(seg)
    
    # Find final segments (segments with no downstream)
    # A segment has no downstream if its top_node has no parent, or its top_node's parent is not in any other segment
    final_segments = []
    for seg in segments:
        if seg.nodes:
            top_node = seg.nodes[-1]
            # If top_node has no parent, this segment has no downstream
            if not top_node.parent_node:
                final_segments.append(seg)
            else:
                # Check if top_node's parent is in any other segment
                parent = top_node.parent_node
                parent_in_other_segment = False
                for other_seg in segments:
                    if other_seg != seg and other_seg.nodes and parent in other_seg.nodes:
                        parent_in_other_segment = True
                        break
                # If parent is not in any other segment, this segment has no downstream
                if not parent_in_other_segment:
                    final_segments.append(seg)
    
    # #region agent log
    with open(log_path, 'a') as f:
        root_segments_info = [{"index":segments.index(s),"nodes":[(n.plan_id,n.operator_type) for n in s.nodes],"pipeline_latencies":s.pipeline_latencies,"upstream_count":len(s.upstream_segments)} for s in root_segments]
        final_segments_info = [{"index":segments.index(s),"nodes":[(n.plan_id,n.operator_type) for n in s.nodes],"pipeline_latencies":s.pipeline_latencies,"upstream_count":len(s.upstream_segments)} for s in final_segments]
        all_segments_info = [{"index":i,"nodes":[(n.plan_id,n.operator_type) for n in s.nodes],"upstream_count":len(s.upstream_segments),"upstream_indices":[segments.index(us) for us in s.upstream_segments]} for i,s in enumerate(segments)]
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"pdg_builder.py:278","message":"root and final segments found","data":{"root_segments_count":len(root_segments),"root_segments":root_segments_info,"final_segments_count":len(final_segments),"final_segments":final_segments_info,"all_segments":all_segments_info},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    # If multiple root segments, create a virtual final segment (virtual root) that sums all root segment latencies
    # Final segment = virtual root segment when there are multiple root segments
    if len(root_segments) > 1:
        # Sum all root segment latencies
        total_latency = 0.0
        for seg in root_segments:
            # Calculate segment latency: max for cross-stage, sum for inner-stage
            if seg.is_inner_stage:
                seg_latency = sum(seg.pipeline_latencies)
            else:
                seg_latency = max(seg.pipeline_latencies) if seg.pipeline_latencies else 0.0
            total_latency += seg_latency
        
        # Create a virtual final segment (virtual root) that aggregates all root segments
        # This is an inner-stage segment that sums latencies from its upstream root segments
        final_segment = Segment(
            pipeline_latencies=[total_latency],  # Single latency value: sum of all root segments
            is_inner_stage=True,  # Inner-stage segment that aggregates root segments
            nodes=[],  # Empty nodes list
            dop_info={},  # Empty dop_info
            pipelines=[]
        )
        # Set upstream_segments to root_segments so eval_segment can recursively process them
        final_segment.upstream_segments = root_segments.copy()
        
        # #region agent log
        with open(log_path, 'a') as f:
            final_seg_id = id(final_segment)
            root_seg_info = [{"index":segments.index(seg),"nodes":[(n.plan_id,n.operator_type) for n in seg.nodes],"pipeline_latencies":seg.pipeline_latencies} for seg in root_segments]
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"pdg_builder.py:299","message":"virtual final segment created (multiple root segments, sum latencies)","data":{"final_seg_id":final_seg_id,"total_latency":total_latency,"pipeline_latencies":final_segment.pipeline_latencies,"root_segments_count":len(root_segments),"root_segments":root_seg_info,"upstream_count":len(final_segment.upstream_segments),"is_inner_stage":final_segment.is_inner_stage},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        
        return final_segment
    elif len(root_segments) == 1:
        # For sequential dependency chain, return the final segment (not root segment)
        # so that eval_segment can recursively process upstream segments
        if len(final_segments) == 1:
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"pdg_builder.py:301","message":"single final segment returned","data":{"pipeline_latencies":final_segments[0].pipeline_latencies,"upstream_count":len(final_segments[0].upstream_segments)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            return final_segments[0]
        else:
            # Fallback: return root segment if no final segment found
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"pdg_builder.py:301","message":"single root segment returned (no final segment)","data":{"pipeline_latencies":root_segments[0].pipeline_latencies,"upstream_count":len(root_segments[0].upstream_segments)},"timestamp":int(__import__('time').time()*1000)}) + '\n')
            # #endregion
            return root_segments[0]
    else:
        # #region agent log
        with open(log_path, 'a') as f:
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"pdg_builder.py:304","message":"fallback empty segment","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n')
        # #endregion
        # Fallback: create empty segment
        return Segment(pipeline_latencies=[0.0], is_inner_stage=True, pipelines=[])


def convert_stage_dag_to_pdg(thread_blocks: Dict[int, ThreadBlock], all_nodes: List[PlanNode]) -> Segment:
    """
    Convert Stage DAG (ThreadBlocks) to Segment structure.
    Build segments from bottom to top (from leaf nodes upward to breakers).
    
    A segment is either:
    - Inner-stage segment: a single pipeline within one stage
    - Cross-stage segment: adjacent pipelines from different stages
    
    Args:
        thread_blocks: Dict mapping thread_id to ThreadBlock (Stage DAG)
        all_nodes: All nodes in the query plan
        
    Returns:
        Top-level Segment for latency calculation
    """
    import json
    log_path = '/home/zhy/opengauss/tools/new_serverless_predictor/.cursor/debug.log'
    top_segment = build_segments_from_bottom_up(all_nodes, thread_blocks)
    # #region agent log
    with open(log_path, 'a') as f:
        top_seg_id = id(top_segment)
        upstream_ids = [id(us) for us in top_segment.upstream_segments] if top_segment.upstream_segments else []
        f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":"D","location":"pdg_builder.py:422","message":"convert_stage_dag_to_pdg returning","data":{"top_seg_id":top_seg_id,"upstream_count":len(top_segment.upstream_segments) if top_segment.upstream_segments else 0,"upstream_ids":upstream_ids,"self_in_upstream":top_seg_id in upstream_ids},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    return top_segment
