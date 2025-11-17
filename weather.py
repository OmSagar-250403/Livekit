from dotenv import load_dotenv
import os
import aiohttp

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, function_tool
from livekit.plugins import noise_cancellation, silero, deepgram, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool()
    async def lookup_weather(
            self,

            city_name: str,
            country_code: str = "",
    ) -> dict[str, any]:
        """Look up weather information for a given location.

        Args:
            city_name: The location to look up weather information for.
            country_code: Optional country code (e.g., 'IN' for India).
        """

        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            return {"error": "Missing API key"}

        if country_code:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name},{country_code}&appid={api_key}"
        else:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    return {"error": f"Failed to fetch weather data"}
                data = await response.json()

        temp = data["main"]["temp"] - 273.15
        weather_main = data["weather"][0]["main"]
        weather_desc = data["weather"][0]["description"]
        location_name = data["name"]

        return {
            "location": location_name,
            "temperature_celsius": temp,
            "weather_main": weather_main,
            "weather_report": weather_desc,
        }


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STTv2(
            model="flux-general-en",
            eager_eot_threshold=0.4,
        ),
        llm=google.LLM(
            model="gemini-2.0-flash-exp",
        ),
        tts=deepgram.TTS(model="aura-asteria-en",),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
