__all__ = ['SlaveNotReady']

class SlaveNotReady(Exception):
    """Raise when attempting to give a non-ready slave a job"""