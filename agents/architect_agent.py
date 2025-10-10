from agents.data_collector import DataCollectorAgent
from agents.chart_creator import ChartCreatorAgent
from agents.insight_generator import InsightGeneratorAgent

class ArchitectAgent:
    def __init__(self):
        self.data_agent = DataCollectorAgent()
        self.chart_agent = ChartCreatorAgent(self.data_agent)
        self.insight_agent = InsightGeneratorAgent(self.data_agent)