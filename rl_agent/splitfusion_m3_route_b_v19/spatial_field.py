"""Exact approved scalar position field; no time, RNG, policy or outcome input."""
import math


def field(x, y):
    x, y = float(x), float(y)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError('non-finite ego position')
    rho = math.sqrt(((x + 55.0)/80.0)**2 + ((y - 140.0)/45.0)**2)
    u = 0.5 + 0.5*math.sin(2*math.pi*x/180 + 0.4)*math.cos(2*math.pi*y/160 - 0.7)
    high = 23.5 - 3.5*u**3
    low = 13 + math.sin(2*math.pi*(x + 55.0)/160)
    t = max(0.0, min(1.0, (rho - 1.0)/0.25))
    weight = 1 - t*t*(3 - 2*t)
    target = weight*low + (1 - weight)*high
    region = 'LOW_CORE' if rho <= 1 else 'TRANSITION' if rho < 1.25 else 'HIGH'
    if not 12 <= target <= 23.5:
        raise ValueError('approved spatial field bound violated')
    return dict(ego_x=x, ego_y=y, rho=rho, region=region, low_weight=weight, target_snr_db=target)
