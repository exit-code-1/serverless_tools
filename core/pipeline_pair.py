"""
Segment class for segment-based query latency prediction.
Represents a segment (pipeline or pipeline pair) in the query plan.
"""


class Segment:
    """
    Represents a segment in the query plan.
    A segment can contain one or more pipelines (e.g., build and probe for agg/join).
    For cross-stage segments, it can contain pipelines from multiple stages.
    """
    
    def __init__(self, pipeline_latencies=None, is_inner_stage=True, nodes=None, dop_info=None, pipelines=None):
        """
        Initialize a PipelinePair.
        
        Args:
            pipeline_latencies: List of pipeline latencies t(p) for pipelines in this pair
            is_inner_stage: True if inner-stage pipeline pair, False if cross-stage pipeline pair
            nodes: List of PlanNode objects involved in this pipeline pair (for cross-stage pairs)
            dop_info: Dictionary storing DOP information for each stage involved
                     Format: {stage_id: {'dop': int, 'nodes': [PlanNode]}}
            pipelines: List of Pipeline objects in this pair (can span multiple stages for cross-stage pairs)
        """
        self.pipeline_latencies = pipeline_latencies if pipeline_latencies else []
        self.is_inner_stage = is_inner_stage
        self.nodes = nodes if nodes else []
        self.dop_info = dop_info if dop_info else {}
        self.pipelines = pipelines if pipelines else []  # List of Pipeline objects (can span multiple stages)
        self.upstream_segments = []  # List of upstream Segment objects
    
    def add_upstream_segment(self, upstream_segment):
        """Add an upstream segment dependency."""
        # Prevent adding self as upstream
        if upstream_segment is self:
            return
        if upstream_segment not in self.upstream_segments:
            self.upstream_segments.append(upstream_segment)
    
    def add_node(self, node):
        """Add a node to this segment."""
        if node not in self.nodes:
            self.nodes.append(node)
    
    def add_dop_info(self, stage_id, dop, nodes):
        """
        Add DOP information for a stage.
        
        Args:
            stage_id: Identifier for the stage
            dop: Degree of parallelism for this stage
            nodes: List of PlanNode objects in this stage
        """
        self.dop_info[stage_id] = {
            'dop': dop,
            'nodes': nodes
        }
    
    def get_all_nodes(self):
        """Get all nodes involved in this segment."""
        all_nodes = list(self.nodes)
        for stage_info in self.dop_info.values():
            all_nodes.extend(stage_info['nodes'])
        # Remove duplicates while preserving order
        seen = set()
        unique_nodes = []
        for node in all_nodes:
            if id(node) not in seen:
                seen.add(id(node))
                unique_nodes.append(node)
        return unique_nodes
    
    def get_all_dops(self):
        """Get all DOP values involved in this segment."""
        dops = set()
        for stage_info in self.dop_info.values():
            dops.add(stage_info['dop'])
        # Also check nodes directly in the pipeline pair
        for node in self.nodes:
            if hasattr(node, 'dop'):
                dops.add(node.dop)
        # Also check pipelines (which may span multiple stages)
        for pipeline in self.pipelines:
            # Get DOP from pipeline's nodes
            for node in pipeline.nodes:
                if hasattr(node, 'dop'):
                    dops.add(node.dop)
        return list(dops)
    
    def get_stage_ids(self):
        """Get all stage IDs involved in this segment."""
        stage_ids = set()
        # From dop_info
        stage_ids.update(self.dop_info.keys())
        # From pipelines (which may span multiple stages for cross-stage pairs)
        for pipeline in self.pipelines:
            stage_ids.add(pipeline.stage_id)
        return list(stage_ids)
