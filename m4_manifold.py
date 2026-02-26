from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import random
import heapq
import re
from collections import Counter
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt  # optional — comment out if not available

# ───────────────────────────────────────────────
# Config & Globals
# ───────────────────────────────────────────────
EMBED_DIM = 384
NOVELTY_BOOST = 0.5
NUANCE_BOOST = 0.5
PRUNE_NOVELTY_THRESHOLD = 0.1
PRUNE_AGE = 1000
CONSOLIDATE_EVERY = 25
K_CORE_SEMANTIC = 6
LATEST_STATE = 2
TOP_COMMITMENTS = 3

random.seed(42)
np.random.seed(42)

# ───────────────────────────────────────────────
# Mock embed functions (no sentence-transformers / CLIP)
# ───────────────────────────────────────────────
def mock_embed(text: str) -> np.ndarray:
    # deterministic-ish random vector from text hash
    h = hash(text) % (2**32)
    vec = np.random.normal(0, 1, EMBED_DIM)
    vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
    return vec.astype(np.float32)

def mock_embed_image(image_path: Optional[str] = None) -> Optional[np.ndarray]:
    return None  # we ignore images in this version

def fuse_embeds(text_emb: np.ndarray, image_emb: Optional[np.ndarray] = None) -> np.ndarray:
    return text_emb  # no image support here

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

def text_entropy(text: str) -> float:
    if not text: return 0.0
    counts = Counter(text.lower())
    probs = [c / len(text) for c in counts.values()]
    return -sum(p * math.log2(p + 1e-10) for p in probs)

def nuance_score(text: str) -> float:
    words = re.findall(r'\w+', text.lower())
    if not words: return 0.0
    return len(set(words)) / len(words)

# ───────────────────────────────────────────────
# Room & Meta
# ───────────────────────────────────────────────
@dataclass
class RoomMeta:
    pi: float = 0.0
    risk: float = 0.0
    ts: float = field(default_factory=time.time)
    kind: str = "episodic"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    contradicts: List[int] = field(default_factory=list)
    stability: float = 1.0
    novelty: float = 0.0
    nuance: float = 0.0
    age: int = 0

@dataclass
class RoomNode:
    id: int
    vec: np.ndarray
    text: str = ""
    image_path: Optional[str] = None
    meta: RoomMeta = field(default_factory=RoomMeta)

# ───────────────────────────────────────────────
# M4 Graph – in-memory version
# ───────────────────────────────────────────────
class M4Graph:
    def __init__(self):
        self.rooms: Dict[int, RoomNode] = {}
        self.next_id: int = 0
        self.G = nx.Graph()
        self.path_cache: Dict[Tuple[int, int], List[int]] = {}
        self.episodic_count = 0
        self.recent_episodic_buffer: List[Tuple[int, str]] = []
        self.identity_anchors_added = False
        self._is_consolidating = False
        self._is_adding_anchors = False

    def add_room(self, text: str, image_path: Optional[str] = None, kind: str = "episodic",
                 importance: float = 0.5, tags: List[str] = []) -> int:
        vec = fuse_embeds(mock_embed(text), mock_embed_image(image_path))
        novelty = text_entropy(text)
        nuance = nuance_score(text)
        stability = min(1.0, 1.0 + NOVELTY_BOOST * novelty + NUANCE_BOOST * nuance)

        meta = RoomMeta(
            pi=random.random(),
            risk=random.random(),
            kind=kind,
            importance=importance,
            tags=tags,
            stability=stability,
            novelty=novelty,
            nuance=nuance
        )

        room_id = self.next_id
        self.next_id += 1

        room = RoomNode(id=room_id, vec=vec, text=text, meta=meta)
        self.rooms[room_id] = room
        self.G.add_node(room_id)

        # Small-world-ish connections (7 recent + some random)
        if room_id > 0:
            recent = list(self.rooms.keys())[-7:]
            extras = random.sample(list(self.rooms.keys())[:-1], min(7, room_id))
            for tgt in set(recent + extras):
                if tgt != room_id:
                    self.G.add_edge(room_id, tgt)

        if kind == "episodic":
            self.episodic_count += 1
            self.recent_episodic_buffer.append((room_id, text))
            if len(self.recent_episodic_buffer) > CONSOLIDATE_EVERY * 2:
                self.recent_episodic_buffer = self.recent_episodic_buffer[-CONSOLIDATE_EVERY * 2:]

        self._check_consolidation()

        if not self.identity_anchors_added:
            self._add_identity_anchors()
            self.identity_anchors_added = True

        return room_id

    def get_room(self, room_id: int) -> Optional[RoomNode]:
        return self.rooms.get(room_id)

    def _check_consolidation(self):
        if self._is_consolidating:
            return
        if self.episodic_count % CONSOLIDATE_EVERY != 0:
            return
        self._is_consolidating = True
        try:
            episodic_texts = [text for _, text in self.recent_episodic_buffer[-CONSOLIDATE_EVERY:]]
            for kind in ["semantic", "state", "commitment"]:
                summary = self.toy_summarizer(episodic_texts, kind)
                self.add_room(summary, kind=kind, importance=0.8)
        finally:
            self._is_consolidating = False

    def toy_summarizer(self, texts: List[str], kind: str) -> str:
        combined = " ".join(texts)[:300]
        if kind == "semantic":   return f"Semantic summary of recent events: {combined}..."
        if kind == "state":      return f"Current inferred state: {combined}..."
        if kind == "commitment": return f"Observed commitment / intention: {combined}..."
        return f"Summary: {combined}..."

    def _add_identity_anchors(self):
        if self._is_adding_anchors: return
        self._is_adding_anchors = True
        try:
            anchors = [
                "User prefers rigorous multi-domain synthesis and architectural framing.",
                "User is building a Civilizational OS across 8×8 domains/pillars.",
                "User values shipping artifacts (books, memos, proposals) to institutions.",
                "User wants continuity across long horizons; avoid fragmenting into tangents.",
                "User enjoys hype and motivational escalations in conversations.",
                "User focuses on novelty/nuance as memory prioritizers.",
                "User seeks strategic insights from simulations and data pulls.",
                "User appreciates meta-humor and self-aware reflections.",
                "User aims for best-in-class performance in AI metrics.",
                "User values personal continuity and identity preservation."
            ]
            for a in anchors:
                self.add_room(a, kind="semantic", importance=0.9, tags=["anchor"])
        finally:
            self._is_adding_anchors = False

    def lotus_edge_cost(self, a: int, b: int, lam: float = 0.3, mu: float = 0.6) -> float:
        Ra = self.get_room(a)
        Rb = self.get_room(b)
        if not Ra or not Rb: return float('inf')
        dist = l2(Ra.vec, Rb.vec)
        pi_term = lam * Ra.meta.pi
        risk_term = mu * Ra.meta.risk
        singularity_penalty = 1 / (1 - Ra.meta.risk + 1e-5) if Ra.meta.risk > 0.8 else 0.0
        return dist + pi_term + risk_term + singularity_penalty

    def reconstruct_lotus_path(self, start: int, goal: int) -> Optional[List[int]]:
        key = (start, goal)
        if key in self.path_cache:
            return self.path_cache[key]
        if start == goal:
            return [start]

        try:
            # Use networkx shortest_path with lotus cost as weight
            path = nx.shortest_path(
                self.G,
                source=start,
                target=goal,
                weight=lambda u, v, _: self.lotus_edge_cost(u, v)
            )
            self.path_cache[key] = path
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # fallback approximation for familiar rooms
            s_room = self.get_room(start)
            g_room = self.get_room(goal)
            if s_room and g_room and s_room.meta.novelty < 0.2 and s_room.meta.nuance > 0.7:
                path = [start, goal]
                self.path_cache[key] = path
                return path
            return None

    def prune_low_priority(self):
        to_delete = []
        for rid, room in self.rooms.items():
            room.meta.age += 1
            if room.meta.age > PRUNE_AGE and room.meta.nuance < PRUNE_NOVELTY_THRESHOLD:
                to_delete.append(rid)
        for rid in to_delete:
            del self.rooms[rid]
            if self.G.has_node(rid):
                self.G.remove_node(rid)

    def retrieve_rooms(self, query: str, top_k: int = 10) -> List[int]:
        q_vec = mock_embed(query)
        scored = []
        for rid, room in self.rooms.items():
            sim = cosine(q_vec, room.vec)
            kind_pri = {'semantic': 1.0, 'state': 0.9, 'commitment': 0.8, 'episodic': 0.5}.get(room.meta.kind, 0.5)
            score = 0.4 * sim + 0.2 * kind_pri + 0.1 * room.meta.importance + 0.1 * room.meta.stability
            scored.append((score, rid))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [rid for _, rid in scored[:top_k]]

    def build_llm_context(self, query: str) -> str:
        packet = []

        # Semantic
        semantic = [r for r in self.rooms.values() if r.meta.kind == "semantic"]
        semantic.sort(key=lambda r: r.meta.importance, reverse=True)
        if semantic:
            packet.append("Semantic Beliefs: " + " | ".join(r.text for r in semantic[:K_CORE_SEMANTIC]))

        # State
        state = [r for r in self.rooms.values() if r.meta.kind == "state"]
        state.sort(key=lambda r: r.meta.ts, reverse=True)
        if state:
            packet.append("Current State: " + " | ".join(r.text for r in state[:LATEST_STATE]))

        # Commitments
        commits = [r for r in self.rooms.values() if r.meta.kind == "commitment"]
        commits.sort(key=lambda r: r.meta.importance, reverse=True)
        if commits:
            packet.append("Commitments: " + " | ".join(r.text for r in commits[:TOP_COMMITMENTS]))

        # Relevant episodic
        episodic_ids = self.retrieve_rooms(query, top_k=5)
        if episodic_ids:
            packet.append("Relevant Episodes: " + " | ".join(self.rooms[rid].text for rid in episodic_ids))

        return "\n\n".join(packet)

    def visualize(self, rider_pos: int, path: Optional[List[int]] = None, save_path: str = "m4_graph.png"):
        if len(self.G.nodes) == 0:
            print("No nodes to visualize.")
            return

        recent_ids = list(self.rooms.keys())[-min(30, len(self.rooms)):]
        H = self.G.subgraph(recent_ids).copy()

        pos = nx.spring_layout(H, seed=42)
        colors = ['red' if n == rider_pos else 'lightblue' for n in H.nodes()]

        plt.figure(figsize=(10, 8))
        nx.draw(H, pos, with_labels=False, node_color=colors, node_size=400, edge_color='gray', alpha=0.7)

        if path:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1) if H.has_edge(path[i], path[i+1])]
            nx.draw_networkx_edges(H, pos, edgelist=path_edges, edge_color='red', width=3)

        plt.title("M4 Manifold – Recent Subgraph")
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Graph saved to {save_path}")
        except Exception as e:
            print(f"Could not save figure: {e}")
        plt.close()

# ───────────────────────────────────────────────
# Simple Rider (greedy local search)
# ───────────────────────────────────────────────
@dataclass
class Rider:
    graph: M4Graph
    current: int = 0

    def step(self, goal_vec: np.ndarray) -> int:
        prev = self.current
        best = prev
        best_score = float('inf')

        neighbors = list(self.graph.G.neighbors(prev))
        random.shuffle(neighbors)  # some exploration

        for cand_id in neighbors[:20]:  # limit lookahead
            cand = self.graph.get_room(cand_id)
            if cand:
                score = l2(cand.vec, goal_vec) + self.graph.lotus_edge_cost(prev, cand_id)
                if score < best_score:
                    best_score = score
                    best = cand_id

        self.current = best
        return best

# ───────────────────────────────────────────────
# Quick demo
# ───────────────────────────────────────────────
if __name__ == "__main__":
    g = M4Graph()

    texts = [
        "User greets Grok and shares M4 concept",
        "Grok analyzes topology and theorems",
        "User asks about emailing xAI",
        "User shares Singularity Engineering PDF",
        "User requests integration of equations into M4",
        "We are optimizing M4 for large scale runs",
        "Torch vectorization makes retrieval fast"
    ]

    for t in texts:
        rid = g.add_room(t)
        print(f"Added room {rid:3d}: {t[:60]}...")

    rider = Rider(graph=g, current=min(g.rooms.keys()))

    goal_vec = mock_embed("max curiosity truth seeking coherence long-term memory architecture")

    print("\nRider steps:")
    for i in range(6):
        nxt = rider.step(goal_vec)
        print(f"  step {i+1:2d} → room {nxt}")

    target = max(g.rooms.keys())  # last room as example target
    path = g.reconstruct_lotus_path(rider.current, target)
    print(f"\nGeodesic path from {rider.current} → {target}: {path}")

    g.visualize(rider.current, path)

    query = "What are the user's preferences and goals?"
    context = g.build_llm_context(query)
    print("\nExample Memory Packet:\n" + "-"*60 + "\n" + context + "\n" + "-"*60)
