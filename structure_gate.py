# structure_gate.py
def should_enable_gcn(struct_sim: float, threshold: float = 0.75) -> bool:
    return struct_sim >= threshold
