from dotenv import load_dotenv
import os
import logging

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.llm import function_tool
from livekit.plugins import (noise_cancellation, silero, deepgram, google)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from eventregistry import EventRegistry, QueryArticlesIter
from newsdataapi import NewsDataApiClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis
            , asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.""",
        )

    @function_tool()
    async def get_news_summary(self):
        """Get general news headlines and daily news updates"""

        try:
            er = EventRegistry(apiKey=os.getenv("EVENT_REGISTRY_API_KEY"))

            q = QueryArticlesIter(lang="eng")

            articles = []
            for art in q.execQuery(er, sortBy="date", maxItems=3):
                title = art.get('title', 'No title')
                articles.append(title)
                if len(articles) >= 3:
                    break

            logger.info(f"EventRegistry found {len(articles)} articles")

            if articles:
                return "Recent news: " + ". ".join(articles)
            else:
                return "No news found."

        except Exception as e:
            logger.error(f"EventRegistry error: {e}")
            return "Unable to fetch news at the moment."

    @function_tool()
    async def get_factual_news(self, query: str):
        """Get factual or historical news data using the NewsData archive API."""
        try:
            logger.info(f"Fetching factual data for query: {query}")
            api = NewsDataApiClient(apikey=os.getenv("NEWSDATA_API_KEY"))

            # fetch factual (archived) data for US, IN, and UK
            response = api.archive_api(q=query, country=["us", "in", "uk"])

            articles = []
            if response and "results" in response:
                for article in response["results"][:3]:
                    title = article.get("title", "No title")
                    source = article.get("source_id", "Unknown source")
                    articles.append(f"{title} ({source})")

            if articles:
                return "Factual data: " + ". ".join(articles)
            else:
                return f"No factual data found about {query}."

        except Exception as e:
            logger.error(f"Error fetching factual data: {e}")
            return "Sorry, I couldn’t retrieve factual data right now."


async def entrypoint(ctx: agents.JobContext):
    logger.info("Starting LiveKit agent session")

    session = AgentSession(
        stt=deepgram.STTv2(
            model="flux-general-en",
            eager_eot_threshold=0.4,
        ),
        llm=google.LLM(
            model="gemini-2.0-flash-exp",
        ),
        tts=deepgram.TTS(model="aura-asteria-en", ),
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
        instructions="Greet the user and let them know you can provide both general news and real-time updates."
    )

    logger.info("Agent session started successfully")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
