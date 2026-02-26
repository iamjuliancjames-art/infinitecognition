# ADD at top (after imports)
import torch
import networkx as nx
from torch.utils.data import DataLoader  # for future batching

# NEW CONFIG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEGREE = 14          # 7 recent + 7 random = classic murmuration
USE_TORCH = True
MAX_CACHE_SIZE = 5000

# In M4Graph __init__ add:
        self.G = nx.Graph()                    # ← the missing graph!
        self.vec_tensor = None                 # will hold torch stack
        self.id_to_idx = {}                    # for fast lookup
        self.path_cache = {}

# REPLACE add_room with this (keeps your real embed logic, adds graph):
    def add_room(self, text: str, image_path: Optional[str] = None, kind: str = "episodic", importance: float = 0.5, tags: List[str] = []) -> int:
        # your existing embed + fuse + meta code...
        text_emb = text_embedder.encode(text)
        image_emb = embed_image(image_path) if image_path else None
        vec_np = fuse_embeds(text_emb, image_emb)
        vec = torch.from_numpy(vec_np).to(DEVICE).float()
        vec /= vec.norm()   # unit vector

        room_id = self.collection.count()
        # ... your chroma add ...

        # === GRAPH FIX ===
        self.G.add_node(room_id)
        self.id_to_idx[room_id] = len(self.id_to_idx)
        if room_id > 0:
            recent = list(self.id_to_idx.keys())[-DEGREE//2:]
            randoms = random.sample(list(self.id_to_idx.keys())[:-1], min(DEGREE//2, room_id))
            for tgt in set(recent + randoms):
                if tgt != room_id:
                    self.G.add_edge(room_id, tgt)   # unweighted for now (fast)

        # rebuild tensor stack only every 100 adds (or use torch.cat incrementally)
        if room_id % 100 == 0 or self.vec_tensor is None:
            self.vec_tensor = torch.stack([torch.from_numpy(self.collection.get(ids=[str(i)])['embeddings'][0]).to(DEVICE) for i in self.id_to_idx])

        self._check_consolidation()
        if not self.identity_anchors_added:
            self._add_identity_anchors()
            self.identity_anchors_added = True
        return room_id

# NEW: fast batched retrieval
    def retrieve_rooms(self, query: str, top_k: int = 10) -> List[int]:
        q_emb = text_embedder.encode(query)
        q_t = torch.from_numpy(q_emb).to(DEVICE).float()
        q_t /= q_t.norm()
        sims = torch.matmul(q_t.unsqueeze(0), self.vec_tensor.T).squeeze(0)
        topk = torch.topk(sims, min(top_k, len(sims)))
        return [int(list(self.id_to_idx.keys())[i]) for i in topk.indices]

# FIXED path recon (now real + cached)
    def reconstruct_lotus_path(self, start: int, goal: int) -> Optional[List[int]]:
        key = (start, goal)
        if key in self.path_cache:
            return self.path_cache[key]
        try:
            path = nx.shortest_path(self.G, start, goal)   # add weight=self.lotus_edge_cost later for full lotus
            if len(self.path_cache) > MAX_CACHE_SIZE:
                self.path_cache.pop(next(iter(self.path_cache)))
            self.path_cache[key] = path
            return path
        except nx.NetworkXNoPath:
            return None
