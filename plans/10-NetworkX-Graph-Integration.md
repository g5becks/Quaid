# 10 - NetworkX Graph Integration

**Advanced knowledge graph capabilities using NetworkX for intelligent relationship analysis and visualization**

---

## Executive Summary

NetworkX integration transforms Quaid's simple relationship storage into a powerful graph engine capable of advanced knowledge analysis, visualization, and intelligent recommendations. This enhancement maintains the git-native JSONL storage while adding sophisticated graph algorithms, centrality analysis, path finding, and interactive visualization capabilities.

**Key Innovation**: Combine NetworkX's graph algorithms with Polars' data processing and Tantivy's search to create a multi-modal knowledge discovery system that can uncover hidden relationships, suggest connections, and provide intelligent navigation through complex knowledge spaces.

---

## Architecture Overview

### Graph Storage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Graph Layer                    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              NetworkX Graph Engine                   │   │
│  │                                                     │   │
│  │  • Directed graph (DiGraph) for relationships       │   │
│  │  • Node attributes from fragment metadata           │   │
│  │  • Edge attributes for relationship details         │   │
│  │  • Multi-algorithm analysis support                │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Graph Analysis Engine                     │   │
│  │                                                     │   │
│  │  • Path finding algorithms                          │   │
│  │  • Centrality and importance scoring               │   │
│  │  • Community detection                             │   │
│  │  • Cycle detection and dependency analysis        │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Storage Layer (JSONL)                  │   │
│  │                                                     │   │
│  │  • Git-native JSONL for persistence                 │   │
│  │  • Atomic operations with FileLock                 │   │
│  │  • Incremental updates and versioning              │   │
│  │  • Backup and migration support                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
Fragment Creation → Relationship Detection → NetworkX Graph → Analysis Results
      ↓                        ↓                      ↓                ↓
Markdown Storage     Auto/Manual Relationships   Graph Operations   TUI/API Views
      ↓                        ↓                      ↓                ↓
Tantivy Index        JSONL Edge Storage     NetworkX Algorithms   Rich Visualizations
```

---

## Core Components

### 1. Enhanced Knowledge Graph Class

```python
import networkx as nx
import polars as pl
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import logging

@dataclass
class Relationship:
    """Represents a relationship between two fragments"""
    from_id: str
    to_id: str
    relationship: str  # implements, references, depends-on, related-to, supersedes
    confidence: float = 1.0
    created: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSONL storage"""
        return {
            'from_id': self.from_id,
            'to_id': self.to_id,
            'relationship': self.relationship,
            'confidence': self.confidence,
            'created': self.created.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class GraphStats:
    """Graph statistics for monitoring and analytics"""
    node_count: int = 0
    edge_count: int = 0
    relationship_types: Dict[str, int] = field(default_factory=dict)
    connected_components: int = 0
    average_path_length: float = 0.0
    density: float = 0.0
    cycles_detected: int = 0

class EnhancedKnowledgeGraph:
    """Advanced knowledge graph using NetworkX with JSONL persistence"""

    # Relationship type definitions with weights
    RELATIONSHIP_WEIGHTS = {
        'implements': 1.0,      # Strong relationship
        'supersedes': 0.9,      # Strong replacement
        'depends-on': 0.8,      # Strong dependency
        'references': 0.6,      # Moderate reference
        'related-to': 0.4       # Weak relationship
    }

    def __init__(self, storage_path: str, fragments_df: Optional[pl.DataFrame] = None):
        self.storage_path = Path(storage_path)
        self.graph_file = self.storage_path / "graph.jsonl"
        self.fragments_df = fragments_df

        # NetworkX graph
        self.graph = nx.DiGraph()
        self._logger = logging.getLogger(__name__)

        # Performance optimization
        self._graph_stats: Optional[GraphStats] = None
        self._centrality_cache: Optional[Dict[str, float]] = None
        self._community_cache: Optional[Dict[str, int]] = None

        # Load existing graph
        self.load_from_storage()

    def load_from_storage(self) -> None:
        """Load graph from JSONL storage"""
        if not self.graph_file.exists():
            self._logger.info("No existing graph file found, starting with empty graph")
            return

        try:
            # Load relationships using Polars for performance
            relationships_df = pl.read_ndjson(self.graph_file)

            # Build NetworkX graph
            self.graph = nx.DiGraph()

            for row in relationships_df.iter_rows(named=True):
                # Convert ISO datetime string back to datetime object
                created = datetime.fromisoformat(row['created']) if row.get('created') else datetime.now()

                self.graph.add_edge(
                    row['from_id'],
                    row['to_id'],
                    relationship=row['relationship'],
                    confidence=row.get('confidence', 1.0),
                    created=created,
                    metadata=row.get('metadata', {}),
                    weight=self.RELATIONSHIP_WEIGHTS.get(row['relationship'], 0.5)
                )

            self._logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

            # Invalidate caches
            self._invalidate_caches()

        except Exception as e:
            self._logger.error(f"Error loading graph from storage: {e}")
            self.graph = nx.DiGraph()

    def save_to_storage(self, atomic: bool = True) -> None:
        """Save graph to JSONL storage with optional atomic operation"""
        try:
            # Convert NetworkX edges to list of dictionaries
            edges_data = []
            for from_node, to_node, data in self.graph.edges(data=True):
                edge_dict = {
                    'from_id': from_node,
                    'to_id': to_node,
                    'relationship': data['relationship'],
                    'confidence': data.get('confidence', 1.0),
                    'created': data.get('created', datetime.now()).isoformat(),
                    'metadata': data.get('metadata', {})
                }
                edges_data.append(edge_dict)

            # Create DataFrame and save
            df = pl.DataFrame(edges_data)

            if atomic:
                # Atomic write using temporary file
                temp_file = self.graph_file.with_suffix('.tmp')
                df.write_ndjson(temp_file)
                temp_file.replace(self.graph_file)
            else:
                df.write_ndjson(self.graph_file)

            self._logger.info(f"Saved graph with {len(edges_data)} edges to storage")

        except Exception as e:
            self._logger.error(f"Error saving graph to storage: {e}")
            raise

    def enrich_with_fragment_metadata(self) -> None:
        """Add fragment metadata as node attributes"""
        if self.fragments_df is None:
            self._logger.warning("No fragments DataFrame available for enrichment")
            return

        for fragment in self.fragments_df.iter_rows(named=True):
            fragment_id = fragment['id']
            if fragment_id in self.graph:
                # Add rich metadata to node
                self.graph.nodes[fragment_id].update({
                    'title': fragment.get('title', ''),
                    'type': fragment.get('type', 'unknown'),
                    'tags': fragment.get('tags', []),
                    'importance': fragment.get('importance', 'medium'),
                    'created': fragment.get('created'),
                    'entities': fragment.get('entities', []),
                    'has_code': fragment.get('has_code', False),
                    'code_languages': fragment.get('code_languages', []),
                    'word_count': fragment.get('word_count', 0),
                    'completeness': fragment.get('completeness', 'draft')
                })

        self._invalidate_caches()
        self._logger.info("Enriched graph nodes with fragment metadata")

    def add_relationship(self, relationship: Relationship) -> bool:
        """Add a relationship to the graph"""
        try:
            self.graph.add_edge(
                relationship.from_id,
                relationship.to_id,
                relationship=relationship.relationship,
                confidence=relationship.confidence,
                created=relationship.created,
                metadata=relationship.metadata,
                weight=self.RELATIONSHIP_WEIGHTS.get(relationship.relationship, 0.5)
            )

            self._invalidate_caches()
            return True

        except Exception as e:
            self._logger.error(f"Error adding relationship: {e}")
            return False

    def remove_relationship(self, from_id: str, to_id: str) -> bool:
        """Remove a relationship from the graph"""
        try:
            if self.graph.has_edge(from_id, to_id):
                self.graph.remove_edge(from_id, to_id)
                self._invalidate_caches()
                return True
            return False

        except Exception as e:
            self._logger.error(f"Error removing relationship: {e}")
            return False

    def _invalidate_caches(self) -> None:
        """Invalidate all cached computations"""
        self._graph_stats = None
        self._centrality_cache = None
        self._community_cache = None
```

### 2. Graph Analysis Algorithms

```python
class GraphAnalyzer:
    """Advanced graph analysis using NetworkX algorithms"""

    def __init__(self, graph: EnhancedKnowledgeGraph):
        self.graph = graph.graph
        self._logger = logging.getLogger(__name__)

    def find_related_concepts(self, fragment_id: str, max_depth: int = 2,
                            relationship_types: Optional[List[str]] = None) -> List[Dict]:
        """Find related concepts using multi-hop relationship analysis"""
        if fragment_id not in self.graph:
            return []

        related_fragments = []

        # Filter by relationship types if specified
        subgraph = self.graph
        if relationship_types:
            edges_to_keep = [
                (u, v, data) for u, v, data in self.graph.edges(data=True)
                if data['relationship'] in relationship_types
            ]
            subgraph = nx.DiGraph()
            subgraph.add_edges_from([(u, v, data) for u, v, data in edges_to_keep])

        # Find nodes within specified depth
        for target, distance in nx.single_source_shortest_path_length(
            subgraph, fragment_id, cutoff=max_depth
        ).items():
            if target != fragment_id:
                # Get relationship path
                try:
                    path = nx.shortest_path(subgraph, fragment_id, target)
                    path_relationships = self._extract_path_relationships(subgraph, path)

                    related_fragments.append({
                        'fragment_id': target,
                        'distance': distance,
                        'path': path,
                        'relationships': path_relationships,
                        'node_data': subgraph.nodes[target]
                    })
                except nx.NetworkXNoPath:
                    continue

        # Sort by distance (closer relationships first)
        related_fragments.sort(key=lambda x: x['distance'])
        return related_fragments

    def _extract_path_relationships(self, graph: nx.DiGraph, path: List[str]) -> List[Dict]:
        """Extract relationship information from a path"""
        relationships = []
        for i in range(len(path) - 1):
            from_node, to_node = path[i], path[i + 1]
            if graph.has_edge(from_node, to_node):
                edge_data = graph.edges[from_node, to_node]
                relationships.append({
                    'from': from_node,
                    'to': to_node,
                    'type': edge_data['relationship'],
                    'confidence': edge_data.get('confidence', 1.0)
                })
        return relationships

    def find_implementation_chains(self, concept_id: str) -> List[Dict]:
        """Find all implementation chains from concept to code"""
        implementation_chains = []

        # Find all implementation nodes
        implementation_nodes = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('type') == 'implementation'
        ]

        for impl_node in implementation_nodes:
            try:
                # Find all paths from concept to implementation
                paths = list(nx.all_simple_paths(
                    self.graph, concept_id, impl_node, cutoff=5
                ))

                for path in paths:
                    chain = {
                        'implementation_id': impl_node,
                        'path_length': len(path) - 1,
                        'path': path,
                        'relationships': self._extract_path_relationships(self.graph, path),
                        'confidence': self._calculate_path_confidence(path)
                    }
                    implementation_chains.append(chain)

            except nx.NetworkXNoPath:
                continue

        # Sort by path length and confidence
        implementation_chains.sort(key=lambda x: (x['path_length'], -x['confidence']))
        return implementation_chains

    def _calculate_path_confidence(self, path: List[str]) -> float:
        """Calculate confidence score for a path based on edge confidences"""
        if len(path) < 2:
            return 1.0

        confidences = []
        for i in range(len(path) - 1):
            from_node, to_node = path[i], path[i + 1]
            if self.graph.has_edge(from_node, to_node):
                edge_data = self.graph.edges[from_node, to_node]
                confidences.append(edge_data.get('confidence', 1.0))

        # Use geometric mean to penalize paths with weak links
        if not confidences:
            return 0.0

        product = 1.0
        for conf in confidences:
            product *= conf

        return product ** (1.0 / len(confidences))

    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality metrics for all nodes"""
        centrality_metrics = {}

        # PageRank - importance based on link structure
        pagerank = nx.pagerank(self.graph, weight='weight')

        # Betweenness centrality - nodes that connect different parts
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')

        # In-degree centrality - how many things reference this node
        in_degree = nx.in_degree_centrality(self.graph)

        # Out-degree centrality - how many things this node references
        out_degree = nx.out_degree_centrality(self.graph)

        # Closeness centrality - how close to all other nodes
        if nx.is_connected(self.graph.to_undirected()):
            closeness = nx.closeness_centrality(self.graph)
        else:
            # For disconnected graphs, calculate for largest component
            largest_cc = max(nx.strongly_connected_components(self.graph), key=len)
            subgraph = self.graph.subgraph(largest_cc)
            closeness = nx.closeness_centrality(subgraph)

        # Combine metrics
        for node in self.graph.nodes():
            centrality_metrics[node] = {
                'pagerank': pagerank.get(node, 0.0),
                'betweenness': betweenness.get(node, 0.0),
                'in_degree': in_degree.get(node, 0.0),
                'out_degree': out_degree.get(node, 0.0),
                'closeness': closeness.get(node, 0.0),
                'combined_score': (
                    pagerank.get(node, 0.0) * 0.3 +
                    betweenness.get(node, 0.0) * 0.3 +
                    in_degree.get(node, 0.0) * 0.2 +
                    out_degree.get(node, 0.0) * 0.1 +
                    closeness.get(node, 0.0) * 0.1
                )
            }

        return centrality_metrics

    def detect_communities(self) -> Dict[str, int]:
        """Detect communities using the Louvain algorithm"""
        try:
            import community as community_louvain

            # Convert to undirected graph for community detection
            undirected = self.graph.to_undirected()
            communities = community_louvain.best_partition(undirected)

            return communities

        except ImportError:
            self._logger.warning("python-louvain not installed, using simple clustering")
            # Fallback: simple clustering by connected components
            communities = {}
            for i, component in enumerate(nx.weakly_connected_components(self.graph)):
                for node in component:
                    communities[node] = i
            return communities

    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies in the graph"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except Exception as e:
            self._logger.error(f"Error detecting cycles: {e}")
            return []

    def find_bridges(self) -> List[Tuple[str, str]]:
        """Find bridge edges whose removal would disconnect the graph"""
        try:
            # Convert to undirected for bridge detection
            undirected = self.graph.to_undirected()
            bridges = list(nx.bridges(undirected))
            return bridges
        except Exception as e:
            self._logger.error(f"Error finding bridges: {e}")
            return []

    def calculate_graph_statistics(self) -> GraphStats:
        """Calculate comprehensive graph statistics"""
        stats = GraphStats()

        stats.node_count = self.graph.number_of_nodes()
        stats.edge_count = self.graph.number_of_edges()

        # Relationship type distribution
        relationship_types = {}
        for _, _, data in self.graph.edges(data=True):
            rel_type = data['relationship']
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        stats.relationship_types = relationship_types

        # Connected components
        stats.connected_components = nx.number_weakly_connected_components(self.graph)

        # Graph density
        stats.density = nx.density(self.graph)

        # Average path length (for largest component)
        if stats.node_count > 1:
            largest_cc = max(nx.weakly_connected_components(self.graph), key=len)
            if len(largest_cc) > 1:
                subgraph = self.graph.subgraph(largest_cc).to_undirected()
                stats.average_path_length = nx.average_shortest_path_length(subgraph)

        # Cycles detected
        stats.cycles_detected = len(self.detect_cycles())

        return stats
```

### 3. TUI Visualization Components

```python
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import DataTable, Static, Tree, ProgressBar
from textual.reactive import reactive
from rich.table import Table
from rich.tree import Tree as RichTree
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import io
import base64

class GraphVisualizationWidget(Container):
    """Interactive graph visualization widget for TUI"""

    def __init__(self, graph: EnhancedKnowledgeGraph):
        super().__init__()
        self.graph = graph
        self.analyzer = GraphAnalyzer(graph)
        self.current_center = reactive("")
        self.visualization_mode = reactive("ascii")  # ascii, table, stats

    def compose(self) -> ComposeResult:
        """Compose the graph visualization interface"""
        with Vertical():
            yield Static("🔗 Knowledge Graph Explorer", classes="header")

            with Horizontal():
                yield Static("Center Node:", classes="label")
                # Add input widget for center node selection

            with Vertical(id="graph-content"):
                yield Static("Select a fragment to explore its relationships", id="graph-display")

            with Horizontal(id="graph-controls"):
                yield Static("[M]ode: ascii | table | stats", classes="help-text")
                yield Static("[C]enter | [D]epth | [R]efresh", classes="help-text")

    def watch_current_center(self, center: str) -> None:
        """React to center node changes"""
        if center:
            self.update_visualization()

    def update_visualization(self) -> None:
        """Update the graph visualization based on current mode and center"""
        if not self.current_center:
            return

        if self.visualization_mode == "ascii":
            self._render_ascii_graph()
        elif self.visualization_mode == "table":
            self._render_relationship_table()
        elif self.visualization_mode == "stats":
            self._render_graph_statistics()

    def _render_ascii_graph(self) -> None:
        """Render graph as ASCII art"""
        try:
            related = self.analyzer.find_related_concepts(
                self.current_center, max_depth=2
            )

            if not related:
                self.query_one("#graph-display", Static).update(
                    f"No relationships found for {self.current_center}"
                )
                return

            # Build ASCII representation
            ascii_graph = self._build_ascii_representation(self.current_center, related)
            self.query_one("#graph-display", Static).update(ascii_graph)

        except Exception as e:
            self.query_one("#graph-display", Static).update(f"Error: {e}")

    def _build_ascii_representation(self, center: str, related: List[Dict]) -> str:
        """Build ASCII representation of the graph"""
        lines = []
        lines.append(f"📊 Knowledge Graph: {center}")
        lines.append("=" * 60)

        # Group by distance
        by_distance = {}
        for item in related:
            distance = item['distance']
            if distance not in by_distance:
                by_distance[distance] = []
            by_distance[distance].append(item)

        # Render center node
        center_data = self.graph.graph.nodes.get(center, {})
        center_title = center_data.get('title', center)[:40]
        center_type = center_data.get('type', 'unknown')
        lines.append(f"🔷 [{center_type.upper()}] {center_title}")
        lines.append("")

        # Render relationships by distance
        for distance in sorted(by_distance.keys()):
            lines.append(f"📍 Distance {distance}:")

            for item in by_distance[distance]:
                node_id = item['fragment_id']
                node_data = item['node_data']
                title = node_data.get('title', node_id)[:35]
                node_type = node_data.get('type', 'unknown')

                # Indentation based on distance
                indent = "  " * distance
                lines.append(f"{indent}🔸 [{node_type}] {title}")

                # Show relationship path
                for rel in item['relationships']:
                    rel_type = rel['type']
                    confidence = rel['confidence']
                    lines.append(f"{indent}  └─[{rel_type}] ({confidence:.1f})")

                lines.append("")

        return "\n".join(lines)

    def _render_relationship_table(self) -> None:
        """Render relationships as a rich table"""
        try:
            table = Table(title=f"Relationships: {self.current_center}")
            table.add_column("Direction", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Fragment", style="white")
            table.add_column("Title", style="yellow")
            table.add_column("Confidence", justify="right", style="magenta")

            # Outgoing relationships
            for _, target, edge_data in self.graph.graph.out_edges(
                self.current_center, data=True
            ):
                target_data = self.graph.graph.nodes.get(target, {})
                table.add_row(
                    "→ OUT",
                    edge_data['relationship'],
                    target,
                    target_data.get('title', 'No title')[:30],
                    f"{edge_data.get('confidence', 1.0):.2f}"
                )

            # Incoming relationships
            for source, _, edge_data in self.graph.graph.in_edges(
                self.current_center, data=True
            ):
                source_data = self.graph.graph.nodes.get(source, {})
                table.add_row(
                    "← IN",
                    edge_data['relationship'],
                    source,
                    source_data.get('title', 'No title')[:30],
                    f"{edge_data.get('confidence', 1.0):.2f}"
                )

            self.query_one("#graph-display", Static).update(table)

        except Exception as e:
            self.query_one("#graph-display", Static).update(f"Error: {e}")

    def _render_graph_statistics(self) -> None:
        """Render graph statistics"""
        try:
            stats = self.analyzer.calculate_graph_statistics()
            centrality = self.analyzer.calculate_centrality_metrics()

            # Get stats for current center node
            center_stats = centrality.get(self.current_center, {})

            stats_text = f"""
📈 Graph Statistics for: {self.current_center}
{'='*50}

🔢 Node Metrics:
  • In-Degree: {self.graph.graph.in_degree(self.current_center)}
  • Out-Degree: {self.graph.graph.out_degree(self.current_center)}
  • PageRank Score: {center_stats.get('pagerank', 0):.4f}
  • Betweenness: {center_stats.get('betweenness', 0):.4f}
  • Combined Importance: {center_stats.get('combined_score', 0):.4f}

🌐 Global Graph Stats:
  • Total Nodes: {stats.node_count}
  • Total Edges: {stats.edge_count}
  • Connected Components: {stats.connected_components}
  • Graph Density: {stats.density:.4f}
  • Average Path Length: {stats.average_path_length:.2f}
  • Cycles Detected: {stats.cycles_detected}

📊 Relationship Types:
"""

            for rel_type, count in stats.relationship_types.items():
                stats_text += f"  • {rel_type}: {count}\n"

            self.query_one("#graph-display", Static).update(stats_text)

        except Exception as e:
            self.query_one("#graph-display", Static).update(f"Error: {e}")

    def action_change_mode(self, mode: str) -> None:
        """Change visualization mode"""
        if mode in ["ascii", "table", "stats"]:
            self.visualization_mode = mode
            self.update_visualization()

    def action_set_center(self, fragment_id: str) -> None:
        """Set the center node for visualization"""
        if fragment_id in self.graph.graph:
            self.current_center = fragment_id
        else:
            self.query_one("#graph-display", Static).update(
                f"Fragment {fragment_id} not found in graph"
            )

class GraphTimelineWidget(Container):
    """Timeline visualization of graph activity"""

    def __init__(self, graph: EnhancedKnowledgeGraph):
        super().__init__()
        self.graph = graph

    def compose(self) -> ComposeResult:
        yield Static("📅 Graph Activity Timeline", classes="header")
        yield DataTable(id="timeline-table")

    def update_timeline(self) -> None:
        """Update timeline with recent graph activity"""
        table = self.query_one("#timeline-table", DataTable)
        table.clear(columns=True)

        table.add_column("Date", style="cyan")
        table.add_column("Time", style="green")
        table.add_column("Action", style="white")
        table.add_column("From", style="yellow")
        table.add_column("To", style="yellow")
        table.add_column("Type", style="magenta")

        # Get recent relationships
        recent_edges = []
        for from_node, to_node, data in self.graph.graph.edges(data=True):
            created = data.get('created')
            if created:
                recent_edges.append((created, from_node, to_node, data))

        # Sort by date (most recent first)
        recent_edges.sort(reverse=True, key=lambda x: x[0])

        # Show last 20 activities
        for created, from_node, to_node, data in recent_edges[:20]:
            table.add_row(
                created.strftime("%Y-%m-%d"),
                created.strftime("%H:%M"),
                "Relationship Added",
                from_node[:15],
                to_node[:15],
                data['relationship']
            )

class GraphAnalyticsWidget(Container):
    """Analytics dashboard for graph insights"""

    def __init__(self, graph: EnhancedKnowledgeGraph):
        super().__init__()
        self.graph = graph
        self.analyzer = GraphAnalyzer(graph)

    def compose(self) -> ComposeResult:
        yield Static("📊 Graph Analytics Dashboard", classes="header")
        yield Static(id="analytics-content")

    def update_analytics(self) -> None:
        """Update analytics dashboard"""
        try:
            # Get analytics data
            stats = self.analyzer.calculate_graph_statistics()
            centrality = self.analyzer.calculate_centrality_metrics()
            communities = self.analyzer.detect_communities()
            cycles = self.analyzer.detect_cycles()
            bridges = self.analyzer.find_bridges()

            # Find most important nodes
            top_nodes = sorted(
                centrality.items(),
                key=lambda x: x[1].get('combined_score', 0),
                reverse=True
            )[:10]

            # Build analytics report
            analytics_text = f"""
🎯 Key Insights
{'='*50}

🌟 Most Important Nodes:
"""

            for node_id, metrics in top_nodes:
                node_data = self.graph.graph.nodes.get(node_id, {})
                title = node_data.get('title', node_id)[:30]
                score = metrics.get('combined_score', 0)
                analytics_text += f"  • {title} (score: {score:.3f})\n"

            analytics_text += f"""
🏘️ Community Structure:
  • Total Communities: {len(set(communities.values()))}
  • Largest Community Size: {max(communities.values(), key=communities.values().count, default=0) + 1}

⚠️ Issues Detected:
  • Circular Dependencies: {len(cycles)}
  • Critical Bridges: {len(bridges)}

📈 Health Indicators:
  • Graph Connectivity: {'Strong' if stats.connected_components == 1 else f'Fragmented ({stats.connected_components} components)'}
  • Graph Density: {'High' if stats.density > 0.1 else 'Low'} ({stats.density:.4f})
  • Average Path Length: {stats.average_path_length:.2f}
"""

            self.query_one("#analytics-content", Static).update(analytics_text)

        except Exception as e:
            self.query_one("#analytics-content", Static).update(f"Analytics Error: {e}")
```

### 4. Graph Export and Visualization

```python
class GraphExporter:
    """Export graph in various formats for external visualization"""

    def __init__(self, graph: EnhancedKnowledgeGraph):
        self.graph = graph
        self.analyzer = GraphAnalyzer(graph)

    def export_graphviz(self, output_path: str, center_node: Optional[str] = None,
                       max_depth: int = 3) -> str:
        """Export graph in Graphviz DOT format"""
        try:
            # Get subgraph if center node specified
            if center_node and center_node in self.graph.graph:
                nodes = nx.single_source_shortest_path_length(
                    self.graph.graph, center_node, cutoff=max_depth
                ).keys()
                subgraph = self.graph.graph.subgraph(nodes)
            else:
                subgraph = self.graph.graph

            # Create DOT content
            dot_content = "digraph KnowledgeGraph {\n"
            dot_content += "  rankdir=LR;\n"
            dot_content += "  node [shape=box, style=filled];\n"
            dot_content += "  edge [fontsize=8];\n\n"

            # Add nodes with styling by type
            for node_id, data in subgraph.nodes(data=True):
                node_type = data.get('type', 'unknown')
                title = data.get('title', node_id).replace('"', '\\"')

                # Color by type
                colors = {
                    'concept': 'lightblue',
                    'implementation': 'lightgreen',
                    'decision': 'lightyellow',
                    'reference': 'lightgray'
                }
                color = colors.get(node_type, 'white')

                dot_content += f'  "{node_id}" [label="{title}", fillcolor="{color}"];\n'

            # Add edges with labels
            for from_node, to_node, data in subgraph.edges(data=True):
                rel_type = data['relationship']
                confidence = data.get('confidence', 1.0)

                dot_content += f'  "{from_node}" -> "{to_node}" [label="{rel_type} ({confidence:.1f})"];\n'

            dot_content += "}\n"

            # Write to file
            with open(output_path, 'w') as f:
                f.write(dot_content)

            return output_path

        except Exception as e:
            raise Exception(f"Error exporting Graphviz: {e}")

    def export_networkx_visualization(self, output_path: str, format: str = 'png',
                                    center_node: Optional[str] = None,
                                    max_depth: int = 3) -> bytes:
        """Generate graph visualization using matplotlib and NetworkX"""
        try:
            # Use matplotlib agg backend for headless operation
            mplstyle.use('default')
            plt.figure(figsize=(14, 10))

            # Get subgraph if needed
            if center_node and center_node in self.graph.graph:
                nodes = nx.single_source_shortest_path_length(
                    self.graph.graph, center_node, cutoff=max_depth
                ).keys()
                subgraph = self.graph.graph.subgraph(nodes)
            else:
                subgraph = self.graph.graph

            # Layout
            pos = nx.spring_layout(subgraph, k=3, iterations=50, seed=42)

            # Node colors by type
            node_colors = []
            for node in subgraph.nodes():
                node_type = subgraph.nodes[node].get('type', 'unknown')
                color_map = {
                    'concept': '#87CEEB',      # Light blue
                    'implementation': '#90EE90', # Light green
                    'decision': '#FFFFE0',     # Light yellow
                    'reference': '#D3D3D3'     # Light gray
                }
                node_colors.append(color_map.get(node_type, '#FFFFFF'))

            # Node sizes by importance
            centrality = self.analyzer.calculate_centrality_metrics()
            node_sizes = []
            for node in subgraph.nodes():
                importance = centrality.get(node, {}).get('combined_score', 0.1)
                node_sizes.append(500 + importance * 2000)  # Scale between 500-2500

            # Draw the graph
            nx.draw(subgraph, pos,
                    node_color=node_colors,
                    node_size=node_sizes,
                    font_size=8,
                    font_weight='bold',
                    arrows=True,
                    arrowsize=20,
                    edge_color='gray',
                    width=1.5,
                    alpha=0.8,
                    with_labels=False)

            # Add labels for important nodes only
            important_nodes = [
                node for node in subgraph.nodes()
                if centrality.get(node, {}).get('combined_score', 0) > 0.05
            ]

            labels = {
                node: subgraph.nodes[node].get('title', node)[:15]
                for node in important_nodes
            }

            nx.draw_networkx_labels(subgraph, pos, labels, font_size=7)

            # Add edge labels for key relationships
            edge_labels = {}
            for u, v, data in subgraph.edges(data=True):
                if data.get('confidence', 1.0) > 0.7:  # Only show strong relationships
                    edge_labels[(u, v)] = data['relationship'][:8]

            nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=6)

            # Add title and legend
            title = f"Knowledge Graph Visualization"
            if center_node:
                title += f" - Center: {center_node}"
            plt.title(title, fontsize=16, fontweight='bold')

            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB',
                          markersize=10, label='Concept'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#90EE90',
                          markersize=10, label='Implementation'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFFE0',
                          markersize=10, label='Decision'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#D3D3D3',
                          markersize=10, label='Reference')
            ]
            plt.legend(handles=legend_elements, loc='upper right')

            plt.tight_layout()

            # Save to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format=format, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            plt.close()

            return buffer.getvalue()

        except Exception as e:
            raise Exception(f"Error generating visualization: {e}")

    def export_gephi(self, output_path: str) -> Tuple[str, str]:
        """Export graph in Gephi-compatible format (nodes and edges CSV)"""
        try:
            base_path = output_path.replace('.csv', '')
            nodes_file = f"{base_path}_nodes.csv"
            edges_file = f"{base_path}_edges.csv"

            # Export nodes
            nodes_data = []
            for node_id, data in self.graph.graph.nodes(data=True):
                nodes_data.append({
                    'Id': node_id,
                    'Label': data.get('title', node_id),
                    'Type': data.get('type', 'unknown'),
                    'Importance': data.get('importance', 'medium'),
                    'Tags': '|'.join(data.get('tags', [])),
                    'Has_Code': data.get('has_code', False),
                    'Word_Count': data.get('word_count', 0)
                })

            nodes_df = pl.DataFrame(nodes_data)
            nodes_df.write_csv(nodes_file)

            # Export edges
            edges_data = []
            for from_node, to_node, data in self.graph.graph.edges(data=True):
                edges_data.append({
                    'Source': from_node,
                    'Target': to_node,
                    'Type': data['relationship'],
                    'Weight': data.get('confidence', 1.0),
                    'Created': data.get('created', datetime.now()).isoformat()
                })

            edges_df = pl.DataFrame(edges_data)
            edges_df.write_csv(edges_file)

            return nodes_file, edges_file

        except Exception as e:
            raise Exception(f"Error exporting Gephi format: {e}")

    def export_json(self, output_path: str, include_analytics: bool = True) -> str:
        """Export graph in JSON format for web visualization"""
        try:
            # Build graph structure
            graph_data = {
                'directed': True,
                'multigraph': False,
                'graph': {
                    'name': 'Quaid Knowledge Graph',
                    'created': datetime.now().isoformat(),
                    'node_count': self.graph.graph.number_of_nodes(),
                    'edge_count': self.graph.graph.number_of_edges()
                },
                'nodes': [],
                'links': []
            }

            # Add nodes
            for node_id, data in self.graph.graph.nodes(data=True):
                node_data = {
                    'id': node_id,
                    'title': data.get('title', node_id),
                    'type': data.get('type', 'unknown'),
                    'importance': data.get('importance', 'medium'),
                    'tags': data.get('tags', []),
                    'entities': data.get('entities', []),
                    'has_code': data.get('has_code', False),
                    'metadata': {
                        k: v for k, v in data.items()
                        if k not in ['title', 'type', 'importance', 'tags', 'entities', 'has_code']
                    }
                }
                graph_data['nodes'].append(node_data)

            # Add edges
            for from_node, to_node, data in self.graph.graph.edges(data=True):
                edge_data = {
                    'source': from_node,
                    'target': to_node,
                    'relationship': data['relationship'],
                    'confidence': data.get('confidence', 1.0),
                    'created': data.get('created', datetime.now()).isoformat(),
                    'weight': data.get('weight', 0.5)
                }
                graph_data['links'].append(edge_data)

            # Add analytics if requested
            if include_analytics:
                centrality = self.analyzer.calculate_centrality_metrics()
                communities = self.analyzer.detect_communities()
                stats = self.analyzer.calculate_graph_statistics()

                # Add centrality metrics to nodes
                for node in graph_data['nodes']:
                    node_id = node['id']
                    if node_id in centrality:
                        node['centrality'] = centrality[node_id]
                    if node_id in communities:
                        node['community'] = communities[node_id]

                graph_data['analytics'] = {
                    'statistics': {
                        'node_count': stats.node_count,
                        'edge_count': stats.edge_count,
                        'connected_components': stats.connected_components,
                        'density': stats.density,
                        'average_path_length': stats.average_path_length,
                        'cycles_detected': stats.cycles_detected,
                        'relationship_types': stats.relationship_types
                    },
                    'communities': len(set(communities.values())),
                    'bridges': len(self.analyzer.find_bridges())
                }

            # Write to file
            with open(output_path, 'w') as f:
                json.dump(graph_data, f, indent=2, default=str)

            return output_path

        except Exception as e:
            raise Exception(f"Error exporting JSON: {e}")
```

---

## MCP Server Integration

### Graph Tools for MCP Server

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import asyncio

# Register graph tools with MCP server
@mcp.tool()
async def graph_show_relationships(fragment_id: str, max_depth: int = 2,
                                 relationship_types: Optional[List[str]] = None) -> dict:
    """Show relationships for a fragment with NetworkX analysis"""
    try:
        graph = EnhancedKnowledgeGraph(".quaid/memory/indexes", fragments_df)
        graph.load_from_storage()
        graph.enrich_with_fragment_metadata()

        analyzer = GraphAnalyzer(graph)

        # Get related concepts
        related = analyzer.find_related_concepts(
            fragment_id, max_depth, relationship_types
        )

        # Get centrality metrics
        centrality = analyzer.calculate_centrality_metrics()
        fragment_centrality = centrality.get(fragment_id, {})

        # Build ASCII representation
        if related:
            widget = GraphVisualizationWidget(graph)
            ascii_graph = widget._build_ascii_representation(fragment_id, related)
        else:
            ascii_graph = f"No relationships found for {fragment_id}"

        return {
            "fragment_id": fragment_id,
            "related_count": len(related),
            "related_fragments": related,
            "centrality_score": fragment_centrality.get('combined_score', 0.0),
            "pagerank": fragment_centrality.get('pagerank', 0.0),
            "betweenness": fragment_centrality.get('betweenness', 0.0),
            "graph_ascii": ascii_graph,
            "in_degree": graph.graph.in_degree(fragment_id),
            "out_degree": graph.graph.out_degree(fragment_id)
        }

    except Exception as e:
        return {
            "error": str(e),
            "fragment_id": fragment_id
        }

@mcp.tool()
async def graph_find_paths(from_id: str, to_id: str, max_paths: int = 5) -> dict:
    """Find all paths between two fragments"""
    try:
        graph = EnhancedKnowledgeGraph(".quaid/memory/indexes", fragments_df)
        graph.load_from_storage()

        analyzer = GraphAnalyzer(graph)

        # Find all simple paths
        paths = list(nx.all_simple_paths(
            graph.graph, from_id, to_id, cutoff=5
        ))[:max_paths]

        # Calculate path details
        path_details = []
        for path in paths:
            relationships = analyzer._extract_path_relationships(graph.graph, path)
            confidence = analyzer._calculate_path_confidence(path)

            path_details.append({
                "path": path,
                "length": len(path) - 1,
                "relationships": relationships,
                "confidence": confidence
            })

        # Sort by confidence and length
        path_details.sort(key=lambda x: (x['length'], -x['confidence']))

        return {
            "from_id": from_id,
            "to_id": to_id,
            "path_count": len(paths),
            "paths": path_details,
            "direct_connection": graph.graph.has_edge(from_id, to_id)
        }

    except nx.NetworkXNoPath:
        return {
            "from_id": from_id,
            "to_id": to_id,
            "path_count": 0,
            "paths": [],
            "direct_connection": False
        }
    except Exception as e:
        return {
            "error": str(e),
            "from_id": from_id,
            "to_id": to_id
        }

@mcp.tool()
async def graph_analytics() -> dict:
    """Get comprehensive graph analytics"""
    try:
        graph = EnhancedKnowledgeGraph(".quaid/memory/indexes", fragments_df)
        graph.load_from_storage()
        graph.enrich_with_fragment_metadata()

        analyzer = GraphAnalyzer(graph)

        # Calculate statistics
        stats = analyzer.calculate_graph_statistics()
        centrality = analyzer.calculate_centrality_metrics()
        communities = analyzer.detect_communities()
        cycles = analyzer.detect_cycles()
        bridges = analyzer.find_bridges()

        # Find top nodes by importance
        top_nodes = sorted(
            centrality.items(),
            key=lambda x: x[1].get('combined_score', 0),
            reverse=True
        )[:10]

        # Find disconnected nodes
        all_nodes = set(graph.graph.nodes())
        largest_cc = max(nx.weakly_connected_components(graph.graph), key=len)
        disconnected = all_nodes - set(largest_cc)

        return {
            "statistics": {
                "node_count": stats.node_count,
                "edge_count": stats.edge_count,
                "connected_components": stats.connected_components,
                "density": stats.density,
                "average_path_length": stats.average_path_length,
                "cycles_detected": stats.cycles_detected,
                "relationship_types": stats.relationship_types
            },
            "top_nodes": [
                {
                    "id": node_id,
                    "title": graph.graph.nodes[node_id].get('title', node_id),
                    "score": metrics.get('combined_score', 0),
                    "type": graph.graph.nodes[node_id].get('type', 'unknown')
                }
                for node_id, metrics in top_nodes
            ],
            "communities": {
                "count": len(set(communities.values())),
                "largest_size": max(communities.values(), key=communities.values().count, default=0) + 1
            },
            "issues": {
                "cycles": cycles[:5],  # First 5 cycles
                "critical_bridges": bridges,
                "disconnected_nodes": list(disconnected)[:10]  # First 10 disconnected
            },
            "health_indicators": {
                "connectivity": "Strong" if stats.connected_components == 1 else f"Fragmented ({stats.connected_components} components)",
                "density_level": "High" if stats.density > 0.1 else "Low",
                "has_cycles": len(cycles) > 0,
                "has_bridges": len(bridges) > 0
            }
        }

    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
async def graph_add_relationship(from_id: str, to_id: str, relationship: str,
                               confidence: float = 1.0, metadata: Optional[Dict] = None) -> dict:
    """Add a relationship between two fragments"""
    try:
        graph = EnhancedKnowledgeGraph(".quaid/memory/indexes", fragments_df)
        graph.load_from_storage()

        # Validate relationship type
        valid_types = ['implements', 'references', 'depends-on', 'related-to', 'supersedes']
        if relationship not in valid_types:
            return {
                "error": f"Invalid relationship type. Must be one of: {valid_types}",
                "relationship": relationship
            }

        # Create relationship
        rel = Relationship(
            from_id=from_id,
            to_id=to_id,
            relationship=relationship,
            confidence=confidence,
            metadata=metadata or {}
        )

        # Add to graph
        success = graph.add_relationship(rel)
        if success:
            graph.save_to_storage()

            return {
                "success": True,
                "from_id": from_id,
                "to_id": to_id,
                "relationship": relationship,
                "confidence": confidence,
                "message": f"Added {relationship} relationship from {from_id} to {to_id}"
            }
        else:
            return {
                "success": False,
                "error": "Failed to add relationship"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "from_id": from_id,
            "to_id": to_id
        }

@mcp.tool()
async def graph_export(format: str, center_node: Optional[str] = None,
                      max_depth: int = 3, output_path: Optional[str] = None) -> dict:
    """Export graph in various formats"""
    try:
        graph = EnhancedKnowledgeGraph(".quaid/memory/indexes", fragments_df)
        graph.load_from_storage()
        graph.enrich_with_fragment_metadata()

        exporter = GraphExporter(graph)

        if format == "graphviz":
            if not output_path:
                output_path = "knowledge_graph.dot"
            result_path = exporter.export_graphviz(output_path, center_node, max_depth)

            return {
                "success": True,
                "format": "graphviz",
                "output_path": result_path,
                "message": f"Graph exported to {result_path}. Use 'dot -Tpng {result_path} -o graph.png' to render."
            }

        elif format in ["png", "svg", "pdf"]:
            if not output_path:
                output_path = f"knowledge_graph.{format}"

            image_data = exporter.export_networkx_visualization(
                output_path, format, center_node, max_depth
            )

            # Save image data
            with open(output_path, 'wb') as f:
                f.write(image_data)

            return {
                "success": True,
                "format": format,
                "output_path": output_path,
                "file_size": len(image_data),
                "message": f"Graph visualization saved to {output_path}"
            }

        elif format == "json":
            if not output_path:
                output_path = "knowledge_graph.json"
            result_path = exporter.export_json(output_path, include_analytics=True)

            return {
                "success": True,
                "format": "json",
                "output_path": result_path,
                "message": f"Graph exported to {result_path} with analytics"
            }

        elif format == "gephi":
            if not output_path:
                output_path = "knowledge_graph.csv"
            nodes_file, edges_file = exporter.export_gephi(output_path)

            return {
                "success": True,
                "format": "gephi",
                "nodes_file": nodes_file,
                "edges_file": edges_file,
                "message": f"Graph exported for Gephi: {nodes_file} and {edges_file}"
            }

        else:
            return {
                "success": False,
                "error": f"Unsupported format: {format}. Supported: graphviz, png, svg, pdf, json, gephi"
            }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "format": format
        }
```

---

## Benefits of NetworkX Integration

### 1. **Advanced Graph Algorithms**
- **Path Finding**: Find all paths between concepts with confidence scoring
- **Centrality Analysis**: Identify important fragments using PageRank, betweenness, and combined metrics
- **Community Detection**: Discover clusters of related knowledge using Louvain algorithm
- **Cycle Detection**: Identify circular dependencies that might indicate issues

### 2. **Rich Visualization Capabilities**
- **ASCII Art**: Terminal-friendly graph visualization for TUI
- **Interactive Tables**: Sortable, filterable relationship tables
- **Statistical Dashboards**: Comprehensive graph analytics and health indicators
- **Export Formats**: Graphviz, PNG/SVG, JSON, Gephi-compatible formats

### 3. **Performance and Scalability**
- **Efficient Algorithms**: NetworkX provides optimized graph operations
- **Lazy Computation**: Centrality and community detection cached until needed
- **Memory Management**: Streaming operations for large graphs
- **Parallel Processing**: Multi-threaded analysis where possible

### 4. **Intelligent Features**
- **Relationship Scoring**: Weight-based importance scoring for relationships
- **Multi-hop Analysis**: Find related concepts across multiple relationship steps
- **Confidence Calculation**: Path confidence based on edge weights
- **Health Monitoring**: Automatic detection of graph issues and metrics

### 5. **Integration Benefits**
- **MCP Server Tools**: Rich graph operations available to AI agents
- **TUI Widgets**: Interactive visualization components
- **JSONL Persistence**: Maintains git-native storage while adding powerful analysis
- **Fragment Metadata**: Enriches graph with fragment properties for better analysis

---

## Implementation Roadmap

### Phase 1: Core NetworkX Integration (Week 1)
- [ ] Install NetworkX and python-louvain dependencies
- [ ] Implement EnhancedKnowledgeGraph class
- [ ] Add JSONL persistence with atomic operations
- [ ] Create basic relationship management methods

### Phase 2: Graph Analysis Engine (Week 2)
- [ ] Implement GraphAnalyzer with centrality algorithms
- [ ] Add path finding and confidence calculation
- [ ] Implement community detection
- [ ] Add cycle detection and bridge analysis

### Phase 3: TUI Integration (Week 3)
- [ ] Create GraphVisualizationWidget for TUI
- [ ] Implement ASCII graph rendering
- [ ] Add relationship table visualization
- [ ] Create analytics dashboard widget

### Phase 4: Export and Visualization (Week 4)
- [ ] Implement GraphExporter with multiple formats
- [ ] Add Graphviz DOT export
- [ ] Create matplotlib-based visualization
- [ ] Add JSON and Gephi export formats

### Phase 5: MCP Server Tools (Week 5)
- [ ] Register graph tools with MCP server
- [ ] Implement relationship management tools
- [ ] Add analytics and path finding tools
- [ ] Create export functionality tools

### Phase 6: Performance Optimization (Week 6)
- [ ] Add caching for expensive operations
- [ ] Implement incremental graph updates
- [ ] Optimize memory usage for large graphs
- [ ] Add performance monitoring and metrics

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.dependencies]
# ... existing dependencies ...
networkx = "^3.2"                    # Graph algorithms and analysis
python-louvain = "^0.16"            # Community detection (optional)
matplotlib = "^3.8"                  # Graph visualization
graphviz = "^0.20"                  # Graphviz integration (optional)
```

Add to development dependencies:

```toml
[project.optional-dependencies]
graph-viz = [
    "graphviz>=0.20",                # System Graphviz required
    "pillow>=10.0"                   # Image processing for exports
]
```

---

## Usage Examples

### Basic Relationship Analysis

```python
# Load and analyze graph
graph = EnhancedKnowledgeGraph(".quaid/memory/indexes", fragments_df)
graph.load_from_storage()
graph.enrich_with_fragment_metadata()

analyzer = GraphAnalyzer(graph)

# Find related concepts
related = analyzer.find_related_concepts("jwt-auth-001", max_depth=2)

# Find implementation chains
implementations = analyzer.find_implementation_chains("jwt-concept-001")

# Get centrality metrics
centrality = analyzer.calculate_centrality_metrics()
jwt_importance = centrality.get("jwt-auth-001", {}).get('combined_score', 0)
```

### TUI Visualization

```python
# Create TUI widget
widget = GraphVisualizationWidget(graph)
widget.current_center = "jwt-auth-001"
widget.visualization_mode = "ascii"
widget.update_visualization()
```

### Export for External Visualization

```python
# Export for Graphviz
exporter = GraphExporter(graph)
dot_file = exporter.export_graphviz("auth_graph.dot", "jwt-auth-001")

# Export as PNG
png_data = exporter.export_networkx_visualization("auth_graph.png", "png", "jwt-auth-001")

# Export for Gephi
nodes_file, edges_file = exporter.export_gephi("knowledge_graph.csv")
```

---

**Previous**: [09-Claude-Self-Reflect-Adaptations.md](09-Claude-Self-Reflect-Adaptations.md) | **Next**: [11-Testing-and-Quality.md](11-Testing-and-Quality.md)