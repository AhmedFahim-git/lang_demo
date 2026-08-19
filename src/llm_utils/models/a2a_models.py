from dataclasses import dataclass, field

from starlette.routing import Route

# class A2ARoutes(BaseModel):
#     agent_card_routes: list[Route]
#     json_rpc_routes: list[Route]


@dataclass
class A2AROUTES:
    agent_card_routes: list[Route] = field(default_factory=list)
    json_rpc_routes: list[Route] = field(default_factory=list)
