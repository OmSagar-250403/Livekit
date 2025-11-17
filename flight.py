from dotenv import load_dotenv
import os
import aiohttp
from amadeus import Client, ResponseError
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.llm import function_tool
from livekit.plugins import noise_cancellation, silero, deepgram, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import Any, Dict

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise and friendly."""
        )
        self._session = None

    @function_tool()
    async def get_flight_position(self, flight_number: str) -> str:
        """
        Retrieve the real-time flight status for a given flight number.

        Args:
            flight_number (str): The flight number (e.g., 'AA100').

        Returns:
            str: A summary of the current flight information .
        """
        api_key = os.getenv("FLIGHT_API_KEY")
        headers = {"x-apikey": api_key}
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://aeroapi.flightaware.com/aeroapi/flights/{flight_number}"
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return f"Unable to retrieve flight information. (HTTP {resp.status})"
                    data: Dict[str, Any] = await resp.json()
            flights = data.get("flights", [])
            if not flights:
                return f"No recent flights found for {flight_number}."
            return str(flights[0])
        except Exception as e:
            return f"Error retrieving flight position: {str(e)}"

    @function_tool()
    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1
    ) -> str:
        """
        Search for available flight.

        Args:
            origin (str): Name of the departure city (e.g., 'New York').
            destination (str): Name of the destination city (e.g., 'London').
            departure_date (str): Departure date in YYYY-MM-DD format.
            adults (int): Number of adult travelers (12+ years). Default 1.

        Returns:
            str: A list of recommended flight offers including airline, price, and times.
        """
        api_key = os.getenv("AMADEUS_API_KEY")
        api_secret = os.getenv("AMADEUS_API_SECRET")

        try:
            amadeus = Client(client_id=api_key, client_secret=api_secret)

            origin_resp = amadeus.reference_data.locations.get(keyword=origin, subType="CITY")
            dest_resp = amadeus.reference_data.locations.get(keyword=destination, subType="CITY")

            if not origin_resp.data or not dest_resp.data:
                return f"Could not find airports for {origin} or {destination}."

            origin_code = origin_resp.data[0]["iataCode"]
            dest_code = dest_resp.data[0]["iataCode"]

            search_response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin_code,
                destinationLocationCode=dest_code,
                departureDate=departure_date,
                adults=adults
            )

            if not search_response.data:
                return f"No flight offers found from {origin} to {destination} on {departure_date}."

            formatted = []
            for i in search_response.data:
                itineraries = i.get("itineraries", [])
                if not itineraries:
                    continue

                segments = itineraries[0].get("segments", [])
                if not segments:
                    continue

                first_seg = segments[0]
                last_seg = segments[-1]
                carrier = first_seg["carrierCode"]
                number = first_seg["number"]
                dep_time = first_seg["departure"]["at"]
                arr_time = last_seg["arrival"]["at"]
                price = i.get("price", {}).get("total", "N/A")

                formatted.append(
                    f"Flight {carrier}{number}: departs {dep_time}, arrives {arr_time}, price {price} EUR"
                )

            if not formatted:
                return f"No detailed flight offers found."

            result = "\n".join(formatted)

            if self._session:
                await self._session.generate_reply(
                    user_input=f"Found {len(formatted)} flight options:\n{result}",
                    instructions="Summarize the best flight options by price and timing clearly."
                )

            return result

        except ResponseError as error:
            return f"Amadeus API error: {error}"

        except Exception as e:
            return f"Unexpected error: {str(e)}"


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STTv2(
            model="flux-general-en",
            eager_eot_threshold=0.4,
        ),
        llm=openai.LLM(model="Qwen3-30B-A3B-Instruct-2507-AWQ-4bit", base_url="http://122.176.24.165:6896/v1"),
        tts=deepgram.TTS(model="aura-asteria-en"),
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