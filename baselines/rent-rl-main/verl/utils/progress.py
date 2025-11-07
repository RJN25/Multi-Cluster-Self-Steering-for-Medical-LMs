"""Progress bar utilities that are safe to use in both main and worker processes."""
from tqdm import tqdm

try:
    # Newer tqdm exposes a specific key error for unknown kwargs
    from tqdm.std import TqdmKeyError
except Exception:
    # Fallback: if not available, define a local alias so we can catch it below
    class TqdmKeyError(Exception):
        pass


def safe_tqdm(*args, **kwargs):
    """Create a tqdm progress bar while being tolerant to kwargs missing in some tqdm builds.
    
    Tries to provide `force=True` to ensure display even in some wrapped envs, and falls
    back to a call without `force` if the tqdm variant rejects it. Also tolerates
    TypeError raised by some wrappers when unexpected kwargs are provided.
    
    This function is safe to use in both the main process and Ray workers.
    """
    # Remove potentially problematic kwargs in Ray worker context
    if kwargs.pop("position", 0) > 0:
        # Nested bars often don't work well in Ray workers
        kwargs.pop("leave", None)
    
    try:
        # Try with force enabled for best behavior in detached terminals
        kwargs_with_force = dict(kwargs)
        kwargs_with_force.setdefault("force", True)
        return tqdm(*args, **kwargs_with_force)
    except (TqdmKeyError, TypeError):
        # Remove 'force' and try again for compatibility
        kwargs = dict(kwargs)
        kwargs.pop("force", None)
        # Some tqdm variants may still raise TypeError for other kwargs; let that bubble up
        return tqdm(*args, **kwargs)