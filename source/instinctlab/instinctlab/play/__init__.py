"""Application-level playback, kept outside the engine adapters.

An adapter only builds and wraps an environment. Viewer selection is a separate application
concern registered here, so an engine library never has to import the playback UI.
"""

from .dispatch import play, register_player
from .env import PlayEnv
from .viser import play_with_viser, visual_meshes_from_mjcf

register_player("isaacsim", "native", "instinctlab.play.isaacsim:play_native")
register_player("isaacsim", "viser", "instinctlab.play.isaacsim:play_viser")
register_player("mjlab", "native", "instinctlab.play.mjlab:play_native")
register_player("mjlab", "viser", "instinctlab.play.mjlab:play_viser")

__all__ = [
    "PlayEnv",
    "play",
    "play_with_viser",
    "register_player",
    "visual_meshes_from_mjcf",
]
