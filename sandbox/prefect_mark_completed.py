# prefect_mark_completed.py

import asyncio
from datetime import datetime, timezone
from prefect import get_client
from prefect.states import Completed

async def find_run_id(name: str):
    async with get_client() as client:
        # grab the most recent 100 runs (tweak limit as needed)
        runs = await client.read_flow_runs(limit=100)
        for run in runs:
            if run.name == name:
                print(f"Found run – name: {run.name}\nid:   {run.id!s}")
                return run.id
        print(f"No flow‐run found with name {name!r}")

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
    run_id = asyncio.run(find_run_id("brawny-cobra"))
    if run_id:
        asyncio.run(mark_completed(run_id))
    else:
        print("No flow run found to mark as completed.")        










if __name__ == "__main__":
    asyncio.run(mark_completed("c776b84a-c76c-400c-8728-b3fc75fe778b"))

