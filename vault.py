# # vault.py
# class Vault:
#     def __init__(self):
#         self.mode = None              # "topic" | "single_doc" | "multi_doc"
#         self.input_data = None        # topic string OR file path(s)
#         self.subtasks = []            # from Pathfinder
#         self.facts = []               # from Harvester: [{text, source}]
#         self.synthesis = None         # from Synthesizer
#         self.final_report = None      # from Scribe
 
#     # def log_fact(self, text, source):
#     #     self.facts.append({"text": text, "source": source})

#     def log_fact(self, text, source):
#         self.facts.append(
#             {
#                 "text": text,
#                 "source": source
#             }
#         )

# vault.py
class Vault:

    def __init__(self):

        # Input
        self.mode = None
        self.input_data = None


        # Agent memory
        self.subtasks = []
        self.facts = []

        # ChromaDB
        self.vector_store = None

        # # Retrieval
        # self.current_query = ""
        # self.retrieved_chunks = []
        # self.retrieval_k = 5

        # Outputs
        self.synthesis = None
        self.final_report = None


    def log_fact(self, text, source):

        self.facts.append(
            {
                "text": text,
                "source": source
            }
        )