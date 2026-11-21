from comfy_api.latest import ComfyExtension

from .xy_nodes import LoulanXYSampler, LoulanXYScheduler

WEB_DIRECTORY = "./web"


class LoulanXY(ComfyExtension):
    async def get_node_list(self):
        return [LoulanXYSampler, LoulanXYScheduler]


async def comfy_entrypoint():
    from . import review  # Register the local results viewer with PromptServer.
    return LoulanXY()
