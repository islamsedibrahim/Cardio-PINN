"""CardioSolv digital twin for Omniverse / Isaac Sim.

``core``, ``ep``, ``mechanics``, ``surrogate`` and ``io`` are pure Python
(NumPy/SciPy/PyTorch); ``usd`` needs ``pxr``; the Kit extension and UI need
``omni.*`` and are only imported inside Kit.
"""

try:  # inside Kit
    import omni.ext  # noqa: F401  (only importable inside Kit)

    from .extension import CardioSolvExtension  # noqa: F401
except ImportError:  # headless / tests
    pass

__version__ = "0.3.0"
