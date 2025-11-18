import os
import platform
from dotenv import load_dotenv
import asyncio
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, function_tool
from livekit.plugins import noise_cancellation, silero, deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from azure.identity import InteractiveBrowserCredential
from msgraph.graph_service_client import GraphServiceClient

load_dotenv(".env")
if platform.system() == "Windows":
    os.environ["LIVEKIT_INFERENCE"] = "false"
    os.environ["LIVEKIT_INFERENCE_USE_SUBPROCESS"] = "false"
    os.environ["LIVEKIT_NO_INFERENCE"] = "true"


class Assistant(Agent):
    token_cache = {}

    def __init__(self, user_id="default_user") -> None:
        super().__init__(instructions="You are a helpful voice AI assistant.")
        tenant_id = os.getenv("OUTLOOK_TENANT_ID")
        client_id = os.getenv("OUTLOOK_CLIENT_ID")
        if user_id in Assistant.token_cache:
            self.credential = Assistant.token_cache[user_id]
        else:
            credential = InteractiveBrowserCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                redirect_uri="http://localhost:8002/"
            )
            Assistant.token_cache[user_id] = credential
            self.credential = credential
        self.graph_client = GraphServiceClient(
            self.credential,
            scopes=["User.Read", "Calendars.Read", "Calendars.ReadWrite"]
        )

    @function_tool()
    async def get_my_calendars(self):
        """Get the list of calendars for the authenticated user."""

        calendars = await self.graph_client.me.calendars.get()
        if not calendars.value:
            return "No calendars found"
        result = ""
        for c in calendars.value:
            result += f"{c.name} | {c.id}\n"
        return result


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STTv2(
            model="flux-general-en", 
            eager_eot_threshold=0.4
        ),

        llm=openai.LLM(
            model="", 
            base_url=""
        ),

        tts=deepgram.TTS(
            model="aura-asteria-en"
        ),

        vad=silero.VAD.load(),
        turn_detection=MultilingualModel()
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    await session.generate_reply(
        instructions="Greet the user and offer assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET")
        )
    )
