import asyncio
from prefect import get_client

async def find_run_id(name: str):
    async with get_client() as client:
        # grab the most recent 100 runs (tweak limit as needed)
        runs = await client.read_flow_runs(limit=100)
        for run in runs:
            if run.name == name:
                print(f"Found run – name: {run.name}\nid:   {run.id!s}")
                return
        print(f"No flow‐run found with name {name!r}")

asyncio.run(find_run_id("fearless-jaguar"))
