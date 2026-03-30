from typing import Dict, List


def to_tribe_command(intent_vector: List[float]) -> Dict[str, object]:
    """
    Translate a continuous intent vector into a high-level TRIBE command.

    The mean energy of the vector determines the command:
      >= 0.65  -> CONFIRM_INTENT  (strong, sustained activation)
      >= 0.40  -> TRANSITION      (rising / ambiguous activation)
      < 0.40   -> IDLE            (resting / no intent detected)
    """
    if not intent_vector:
        return {"command": "IDLE", "energy": 0.0}

    energy = sum(intent_vector) / len(intent_vector)

    if energy >= 0.65:
        command = "CONFIRM_INTENT"
    elif energy >= 0.40:
        command = "TRANSITION"
    else:
        command = "IDLE"

    return {"command": command, "energy": round(energy, 4)}
