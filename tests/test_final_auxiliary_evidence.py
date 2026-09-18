from scripts.audit_final_auxiliary_evidence import transitions


def row(success, status="completed", rejected=False, gated=False):
    trace = [{"type": "verify"}]
    if rejected:
        node = {"type": "refine"}
        if gated:
            node["reserve_rejected"] = True
        trace.append(node)
    return {"success": success, "status": status, "trace": trace}


def test_failed_reference_cannot_be_counted_as_a_repair():
    base = {("valid", 0): row(False), ("failed", 0): row(False, "error:timeout")}
    arm = {key: row(True) for key in base}
    eligible = {key for key, value in base.items() if value["status"] == "completed"}
    result = transitions(base, arm, eligible)
    assert result["pairs"] == result["repairs"] == 1


def test_initial_gating_is_not_execution_and_breakage_paths_partition():
    base = {(name, 0): row(True) for name in ("accept", "execute", "gate")}
    arm = {("accept", 0): row(False), ("execute", 0): row(False, rejected=True),
           ("gate", 0): row(False, rejected=True, gated=True)}
    result = transitions(base, arm, set(base))
    assert result["breakages"] == 3
    assert result["acceptance_breakages"] == 1
    assert result["rejection_breakages"] == 2
    assert result["initial_executed"] == result["initial_gated"] == 1
