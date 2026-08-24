def calculate_complexity(genome: dict) -> int:
    """
    Calculates a complexity score for a given strategy genome.
    Penalizes deeply nested conditions, multiple regimes, and numerous parameters.
    """
    score = 0
    
    direction = genome.get("direction", {})
    if direction:
        score += 1 # Base direction
        params = direction.get("params", {})
        score += len(params)
        
    confirmation = genome.get("confirmation", {})
    if confirmation:
        score += 2 # Confirmation adds more complexity
        score += len(confirmation.keys()) - 1
        
    regime = genome.get("regime", {})
    if regime:
        # Penalize hard regime filtering heavily, or just count the lists
        if "allowed_vol_regimes" in regime:
            score += 2
        if "allowed_trend_regimes" in regime:
            score += 2
            
    sizing = genome.get("sizing", {})
    if sizing:
        score += 1
        score += len(sizing.keys()) - 1
        
    stop = genome.get("stop", {})
    if stop:
        score += 1
        
    # Multi-timeframe context
    mtf = genome.get("multi_timeframe", {})
    if mtf:
        score += 3 # High penalty for MTF
        
    return score
