"""Latent trajectory planning package.

The planner (JEPA + latent continuous thought) predicts the *next K* concept
vectors from context (Task 3), then the writer realizes them into text (Task 4).
"""
from .planner import TrajectoryPlanner

__all__ = ["TrajectoryPlanner"]
