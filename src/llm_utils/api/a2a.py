from llm_utils.db.db_utils import SessionLocal, get_agents, make_agents
from llm_utils.models.a2a_models import A2AROUTES
from llm_utils.services.a2a_service import A2AService

A2A_ROUTES_LIST = []
A2A_SERVICES = []
make_agents()

with SessionLocal() as db_session:
    agents = get_agents(db_session)
    for agent in agents:
        a2a_service = A2AService(
            db_session=SessionLocal(), agent_user_id=agent.agent_user_id
        )
        A2A_SERVICES.append(a2a_service)
        A2A_ROUTES_LIST.append(
            A2AROUTES(
                agent_card_routes=a2a_service.get_agent_card_routes(),
                json_rpc_routes=a2a_service.get_jsonrpc_routes(),
            )
        )
