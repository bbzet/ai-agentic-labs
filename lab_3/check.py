import asyncio
from mcp import Client, StdioServerParameters

server = StdioServerParameters(command="python", args=["server.py"])

async def main():
    async with Client(server) as client:
        tools = await client.list_tools()
        print("Инструменты:", [t.name for t in tools.tools])
        result = await client.call_tool("search_files", {"query": "отпуск"})
        for block in result.content:
            print(block.text)

asyncio.run(main())
