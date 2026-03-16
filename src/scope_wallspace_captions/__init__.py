"""WallSpace Captions — live transcription overlay + prompt forwarding for Daydream Scope."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """Register Pre and Post caption pipelines with Scope."""
    from .pipeline import WallspaceCaptionsPrePipeline, WallspaceCaptionsPostPipeline

    register(WallspaceCaptionsPrePipeline)
    register(WallspaceCaptionsPostPipeline)
