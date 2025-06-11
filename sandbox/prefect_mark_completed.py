import asyncio
from datetime import datetime, timezone

from prefect import get_client
from prefect.states import Completed

async def mark_completed(flow_run_id: str):
    async with get_client() as client:
        # create a Completed state with a timestamp
        completed_state = Completed(timestamp=datetime.now(timezone.utc))
        await client.set_flow_run_state(
            flow_run_id=flow_run_id,
            state=completed_state
        )
        print(f"✅ Flow‐run {flow_run_id} marked as Completed")

if __name__ == "__main__":
    asyncio.run(mark_completed("d3c67e83-3c19-46d7-bd93-2f1ae1e36b46"))
